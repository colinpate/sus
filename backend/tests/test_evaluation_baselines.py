from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from evaluation_baselines import integrate_imu, resample_valid, evaluation_steps, MagneticPowerOracle
from classes.time_series import TimeSeries
from mag_calibration import fit_supervised_power_oracle
from mag_to_travel_model_core import MagToTravelModel
from stats_aggregator import build_mask


class BaselineTests(unittest.TestCase):
    def test_sinusoidal_displacement_units_and_sign(self):
        t = np.arange(0, 40, .005)
        x = .01 * np.sin(2*np.pi*3*t)
        a = -(2*np.pi*3)**2*x
        p = integrate_imu(t,a,cutoff_hz=.3,placement='displacement')
        valid = (t>8)&(t<32)
        e = p[valid]-x[valid]*1000
        self.assertLess(np.std(e), .2)
        self.assertTrue(np.isnan(p[t<2]).all())

    def test_dropout_restarts_and_resampling_does_not_bridge(self):
        t=np.arange(0,30,.01); a=np.sin(t)
        bad=(t>=10)&(t<11)
        p=integrate_imu(t,a,cutoff_hz=1,placement='both',bad=bad)
        changed=a.copy(); changed[t<10]*=100
        q=integrate_imu(t,changed,cutoff_hz=1,placement='both',bad=bad)
        np.testing.assert_allclose(p[t>13],q[t>13],equal_nan=True)
        target=np.arange(0,30,.005)
        res=resample_valid(t,p,target)
        self.assertTrue(np.isnan(res[(target>9)&(target<12)]).all())

    def test_short_missing_samples_do_not_destroy_recording(self):
        t=np.arange(0,20,.005); keep=np.arange(len(t))%100!=0
        p=integrate_imu(t[keep],np.sin(t[keep]),cutoff_hz=1,placement='both')
        self.assertGreater(np.isfinite(p).sum(),2000)

    def test_oracle_reuses_power_family_and_has_offset(self):
        m=np.linspace(100,4000,100)
        y=MagToTravelModel(pred_soft_mg=50).pred_x(m,np.array([200.,6.,.4]))+23
        c,b,e=fit_supervised_power_oracle(m,y,pred_soft_mg=50)
        self.assertLess(e,1e-4)
        np.testing.assert_allclose(MagToTravelModel(pred_soft_mg=50).pred_x(m,c)+b,y,atol=1e-4)

    def test_rear_imu_step_has_no_magnetic_input(self):
        step=evaluation_steps('rear',{})[0]
        self.assertEqual(step.inputs,('accel/lpf/proj','travel'))
        self.assertEqual(len(evaluation_steps('front',{'steps':{'magnetic_power_oracle':{'enabled':False}}})),1)

    def test_common_support_is_shared_by_comparisons(self):
        t=np.arange(10.)
        cache={'active_mask':np.ones(10,dtype=bool)}
        for k in ['travel','travel/solved','travel/mag_model','travel/baseline/accel','travel/oracle/mag_power']:
            cache[k+'__t']=t; cache[k+'__x']=t.copy()
        cache['travel/solved__t']=t+100  # Solver uses a separate time origin.
        cache['travel/baseline/accel__x'][0]=np.nan
        cache['travel/oracle/mag_power__x'][1]=np.nan
        a=build_mask(cache,'travel/solved','travel')
        b=build_mask(cache,'travel/baseline/accel','travel')
        np.testing.assert_array_equal(a,b)
        self.assertEqual(a.sum(),8)

if __name__=='__main__': unittest.main()
