from dataclasses import dataclass

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

import numpy as np

from classes.sensor_loader import Workspace
from classes.time_series import TimeSeries
from classes.step import Step


@dataclass
class GetMagBaseline(Step):
    """Get the baseline for still magnetometer data"""
    still_len_s: float = 0.1 # seconds
    still_a_max: float = 0.5 # m/s^2

    def run(self, ws: Workspace) -> None:
        mag_ts: TimeSeries = ws[self.inputs[0]]
        accel_ts: TimeSeries = ws[self.inputs[1]]

        assert mag_ts.units == "milli-Gauss"
        assert accel_ts.units == "m/s^2"
        still_len = int(self.still_len_s * mag_ts.meta["fs_hz"])
        mag_proj = mag_ts.x
        a_proj_lhp = accel_ts.x
        
        still_mags = []
        for i in range(0, mag_proj.shape[0] - still_len, still_len):
            mag_chunk = mag_proj[i:i+still_len]
            a_chunk = a_proj_lhp[i:i+still_len]
            if max(abs(a_chunk)) < self.still_a_max:
                still_mags.append(mag_chunk)

        mag_baseline = np.median(still_mags) + np.std(still_mags)
        print("Mag baseline", mag_baseline, "std", np.std(still_mags))

        ws[self.outputs[0]] = np.array(mag_baseline)


class FindCalibrationChunks(Step):
    """Find chunks where we can be pretty sure about travel"""
    bump_mag_min = 1000 # mG
    still_a_max: float = 1000 # m/s^2
    bump_dx_min = 20

    still_len_s: float = 0.1 # seconds
    bump_len_s: float = 0.3 # seconds
    stride_s: float = 0.05 # seconds
    skips = 3 # number of following strides to skip if we find a good one, prevents repeats

    def run(self, ws: Workspace) -> None:
        mag_ts: TimeSeries = ws[self.inputs[0]]
        accel_ts: TimeSeries = ws[self.inputs[1]]
        mag_baseline: float = ws[self.inputs[2]]
        mag = mag_ts.x
        accel = accel_ts.x
        t = mag_ts.t
        dt_s = np.diff(t, prepend=t[0]-0.01)

        assert mag_ts.units == "milli-Gauss"
        assert accel_ts.units == "m/s^2"
        still_len = int(self.still_len_s * mag_ts.meta["fs_hz"])
        bump_len = int(self.bump_len_s * mag_ts.meta["fs_hz"])
        stride = int(self.stride_s * mag_ts.meta["fs_hz"])
        chunk_len = still_len + bump_len
        still_mag_max = mag_baseline

        # Find the chunks
        a_mms = accel[:, 0] * 1000
        still_slice = slice(0, still_len)
        bump_slice = slice(still_len, still_len + bump_len)

        slices = []
        a_intint_chunks = []
        i = 0
        skip = 0
        for i in range(0, a_mms.shape[0] - chunk_len, stride):
            if skip > 0:
                skip -= 1
                continue

            chunk_r = slice(i, i+chunk_len)
            chunk_l = slice(i+chunk_len, i, -1)
            chunks = [chunk_r, chunk_l]
            
            for chunk_i in chunks: 
                mag_still = mag[chunk_i][still_slice]
                a_still = a_mms[chunk_i][still_slice]

                a_bump = a_mms[chunk_i][bump_slice]
                dt_bump = dt_s[chunk_i][bump_slice]
                mag_bump = mag[chunk_i][bump_slice]

                mag_still_mean = np.mean(mag_still)

                if np.mean(mag_still) > still_mag_max:
                    continue
                if max(abs(a_still)) > self.still_a_max:
                    continue
                if max(mag_bump) < mag_still_mean + self.bump_mag_min:
                    continue
                #print(f"Found bump at index {i}, still mean mag {mag_still_mean:.1f}, bump max mag {max(mag_bump)}")
                
                a_int = np.cumsum(a_bump * dt_bump)
                print(a_bump.shape, dt_bump.shape, a_int.shape)
                a_intint = np.cumsum(a_int * dt_bump)

                if max(a_intint) < self.bump_dx_min:
                    continue

                skip = self.skips

                a_intint_chunks.append(a_intint)
                slices.append(chunk_i)

        mag_chunks = [mag[chunk_i][bump_slice] for chunk_i in slices]
        
        a_int_chunk_arr = np.array([a_int for a_int in a_intint_chunks])
        mag_chunk_arr = np.array(mag_chunks)
        
        print(a_int_chunk_arr.shape[0], "chunks,", a_int_chunk_arr.shape[1], "samples per chunk")

        ws[self.outputs[0]] = a_int_chunk_arr
        ws[self.outputs[1]] = mag_chunk_arr
        ws[self.outputs[2]] = np.array(slices)