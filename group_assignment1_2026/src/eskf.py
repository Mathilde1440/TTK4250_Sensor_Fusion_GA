import numpy as np

import scipy.linalg
from dataclasses import dataclass, field
from typing import Tuple

from senfuslib import MultiVarGauss
from states import (ImuMeasurement,

                    GnssMeasurement, EskfState)
from states import NominalState, ErrorState

from quaternion import RotationQuaterion
from utils.cross_matrix import get_cross_matrix
from sensors import SensorGNSS
from models import ModelIMU
from solution import eskf as eskf_solu


@dataclass
class ESKF():
    model: ModelIMU
    sensor: SensorGNSS

    def predict_from_imu(self,
                         x_est_prev: EskfState,
                         z_imu: ImuMeasurement,
                         dt: float
                         ) -> EskfState:
        """Method called every time an IMU measurement is received

        Args:
            x_nom_prev: previous eskf state
            z_imu: raw IMU measurement
            dt: time since last imu measurement
        Returns:
            x_est_pred: predicted eskf state
        """
        if dt == 0:
            return x_est_prev

        x_est_prev_nom = x_est_prev.nom

        # z_corr: IMU measurement corrected for bias
        z_corr = self.model.correct_z_imu(x_est_prev_nom, z_imu)

        x_est_pred_nom = self.model.predict_nom(x_est_prev_nom, z_corr, dt)

        x_est_pred_err = self.model.predict_err(x_est_prev, z_corr, dt)

        x_est_pred = EskfState(x_est_pred_nom, x_est_pred_err)

        return x_est_pred

    def update_err_from_gnss(self,
                             x_est_pred: EskfState,
                             z_est_pred: MultiVarGauss[GnssMeasurement],
                             z_gnss: GnssMeasurement
                             ) -> MultiVarGauss[ErrorState]:
        """Update the error state from a gnss measurement

        Hint: see (6.96)
        Due to numerical error its recomended use the robust calculation of 
        posterior covariance, somtimes called Joseph form:
            I_WH = np.eye(*P.shape) - W @ H
            x_err_cov_upd = (I_WH @ P @ I_WH.T + W @ R @ W.T)
        Remember that:
            S = H @ P @ H.T + R
        and that:
            np.linalg.solve(S, H.T) is faster than np.linalg.inv(S)

        Args:
            x_est_pred: predicted nominal and error state (gaussian)
            z_est_pred: predicted gnss measurement (gaussian)
            z_gnss: gnss measurement

        Returns:
            x_est_upd_err: updated error state gaussian
        """
        x_nom = x_est_pred.nom
        x_err = x_est_pred.err
        z_pred, S = z_est_pred

        innovation = z_gnss - z_pred.pos
        H = self.sensor.H(x_nom)
        P = x_err.cov
        R = self.sensor.R
        S = H @ P @ H.T + R
        W = P @ H.T @ np.linalg.solve(S, np.eye(3))
        x_err_upd = np.asarray(x_err.mean) + W @ innovation
        I_WH = np.eye(15) - W @ H
        x_err_cov_upd = I_WH @ P @ I_WH.T + W @ R @ W.T

        x_err_upd = ErrorState.from_array(x_err_upd)
        x_est_upd_err = MultiVarGauss[ErrorState](x_err_upd, x_err_cov_upd)

        return x_est_upd_err

    def inject(self,
               x_est_nom: NominalState,
               x_est_err: MultiVarGauss[ErrorState],
               ) -> EskfState:
        """Perform the injection step

        Hint: see (6.106) and (6.93) on how to inject into nominal state.
        See (6.107) on how to find error covariance after injection

        Args:
            x_nom_prev: previous nominal state
            x_err_upd: updated error state gaussian

        Returns:
            x_est_inj: eskf state after injection
        """
        err = x_est_err.mean
        P = x_est_err.cov

        pos_inj = x_est_nom.pos + err.pos
        vel_inj = x_est_nom.vel + err.vel

        delta_rot = RotationQuaterion.from_avec(err.avec)
        ori_inj = x_est_nom.ori @ delta_rot

        accm_bias_inj = x_est_nom.accm_bias + err.accm_bias
        gyro_bias_inj = x_est_nom.gyro_bias + err.gyro_bias

        x_nom_inj = NominalState(pos_inj, vel_inj, ori_inj,
                                 accm_bias_inj, gyro_bias_inj)

        zero_err = ErrorState(
            pos=np.zeros(3),
            vel=np.zeros(3),
            avec=np.zeros(3),
            accm_bias=np.zeros(3),
            gyro_bias=np.zeros(3)
        )

        E = np.eye(3) - 0.5 * get_cross_matrix(err.avec)
        G_inj = np.eye(15)
        G_inj[6:9, 6:9] = E
        P_inj = G_inj @ P @ G_inj.T

        x_err_inj = MultiVarGauss[ErrorState](zero_err, P_inj)
        x_est_inj = EskfState(x_nom_inj, x_err_inj)
        
        return x_est_inj

    def update_from_gnss(self,
                         x_est_pred: EskfState,
                         z_gnss: GnssMeasurement,
                         ) -> Tuple[NominalState,
                                    MultiVarGauss[ErrorState],
                                    MultiVarGauss]:
        """Method called every time an gnss measurement is received.


        Args:
            x_est_pred: previous estimated esfk state
            z_gnss: gnss measurement

        Returns:
            x_est_upd: updated eskf state
            z_est_upd: predicted measurement gaussian

        """
        z_est_pred = self.sensor.pred_from_est(x_est_pred)
        x_est_upd_err = self.update_err_from_gnss(x_est_pred, z_est_pred, z_gnss)
        x_est_upd = self.inject(x_est_pred.nom, x_est_upd_err)

        return x_est_upd, z_est_pred
