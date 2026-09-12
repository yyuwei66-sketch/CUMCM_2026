"""Focused economic, causal and solver regression tests (standard unittest)."""
import sys
sys.dont_write_bytecode = True
import unittest
import numpy as np
import pandas as pd
from core import (CAP, E0, EMIN, EMAX, ETA, XGRID, Reward, ValueModel, clean_plan, dispatch,
                  execute, features, fit_rewards, historical_future, teacher)


class TerminalValueTests(unittest.TestCase):
    def test_cycle_removal_preserves_grid_state_and_energy(self):
        # 100 charge and 81 discharge change storage by zero; g=19 balances the cycle.
        v=np.array([19.,100.,81.,0.,6000.,6000.,0.])/1000
        plan=clean_plan(v,np.array([0.]),np.array([1.]),1,Reward('none'),'test',0.)
        self.assertAlmostEqual(plan['grid'][0],19.)
        self.assertAlmostEqual(plan['unused'][0],19.)
        self.assertAlmostEqual(plan['charge'][0],0.)
        self.assertAlmostEqual(plan['discharge'][0],0.)
        np.testing.assert_array_equal(plan['storage'],[6000.,6000.])

    def test_one_period_value_includes_discharge_efficiency(self):
        a=dispatch(np.array([10000.]),np.array([1.]),EMIN,dt=1)
        b=dispatch(np.array([10000.]),np.array([1.]),EMIN+1000,dt=1)
        self.assertAlmostEqual(a['objective']-b['objective'],ETA*1000,places=5)

    def test_quadratic_finds_interior_optimum_and_matches_fallback(self):
        net=np.full(24,2000.); p=np.full(24,.5)
        # marginal value crosses charging cost .5/.9 at an interior SOC
        reward=Reward('quadratic',1.,.0001)
        q=dispatch(net,p,EMIN,dt=1,reward=reward)
        fallback=dispatch(net,p,EMIN,dt=1,reward=reward,force_fallback=True)
        expected=EMIN+(1-.5/ETA)/.0001
        self.assertAlmostEqual(q['storage'][-1],expected,delta=.1)
        self.assertAlmostEqual(q['objective'],fallback['objective'],delta=.02)

    def test_zero_curvature_matches_linear(self):
        p=np.r_[np.full(12,.3),np.full(12,1.2)]; net=np.full(24,1000.)
        a=dispatch(net,p,E0,1,Reward('linear',.6))
        b=dispatch(net,p,E0,1,Reward('quadratic',.6,0))
        self.assertAlmostEqual(a['objective'],b['objective'],places=5)

    def test_fallback_preserves_explicit_terminal_constraint(self):
        plan=dispatch(np.full(24,1000.),np.full(24,.5),E0,1,
                      Reward('quadratic',1.,.0001),terminal=4800,force_fallback=True)
        self.assertAlmostEqual(plan['storage'][-1],4800.,places=5)
        with self.assertRaises(ValueError):
            dispatch(np.array([1000.]),np.array([.5]),initial=-1,dt=1)

    def test_pwl_and_linear_are_equivalent_for_linear_values(self):
        p=np.full(24,.5); net=np.full(24,2000.)
        a=dispatch(net,p,E0,1,Reward('linear',.6))
        b=dispatch(net,p,E0,1,Reward('pwl',values=.6*XGRID))
        self.assertAlmostEqual(a['objective'],b['objective'],places=5)

    def test_fitted_values_respect_marginal_constraints(self):
        values=.9*XGRID-.5*.00005*XGRID**2
        quad,lin,err=fit_rewards(values)
        self.assertLess(err,1e-6)
        self.assertAlmostEqual(quad.a,.9,places=7)
        self.assertAlmostEqual(quad.b,.00005,places=9)
        self.assertGreaterEqual(quad.a-quad.b*CAP,0)
        self.assertGreaterEqual(lin.a,0)

    def test_causal_features_labels_and_day_plan(self):
        rng=np.random.default_rng(42)
        data=dict(prices=rng.uniform(.3,1.3,(40,144)),
                  load=rng.uniform(1500,2500,(40,144)),pv=rng.uniform(0,1000,(40,144)))
        d=20
        future=historical_future(data,d); first=teacher(future)
        changed={k:v.copy() for k,v in data.items()}
        for v in changed.values(): v[d:]*=4
        other=historical_future(changed,d); second=teacher(other)
        np.testing.assert_array_equal(first['values'],second['values'])
        np.testing.assert_array_equal(features(future)['econ'],features(other)['econ'])
        # Also exercise the actual inherited Q2 forecast pipeline with mutated future actuals.
        from core import ROOT
        sys.path.insert(0,str(ROOT/'submit'))
        import q2
        dates=pd.date_range('2025-01-01',periods=40)
        s1=q2.forecast_scenarios(dates,data['load'],data['pv'])[2][d]
        s2=q2.forecast_scenarios(dates,changed['load'],changed['pv'])[2][d]
        np.testing.assert_array_equal(s1,s2)
        r1=Reward('quadratic',first['a'],first['b'])
        r2=Reward('quadratic',second['a'],second['b'])
        a=dispatch(np.quantile(s1,.85,axis=0)/6,data['prices'][d-7],reward=r1)
        b=dispatch(np.quantile(s2,.85,axis=0)/6,changed['prices'][d-7],reward=r2)
        np.testing.assert_allclose(a['grid'],b['grid'],atol=1e-5)

    def test_execution_uses_actual_state_and_emergency_last(self):
        ex=execute(np.array([0.,1000.,0.]),np.array([2000.,0.,2000.]),EMIN)
        self.assertEqual(ex['emergency'][0],2000.)
        self.assertGreater(ex['charge'][1],0)
        self.assertGreater(ex['discharge'][2],0)
        np.testing.assert_allclose(np.diff(ex['storage']),ETA*ex['charge']-ex['discharge']/ETA)
        self.assertTrue(((ex['storage']>=EMIN-1e-6)&(ex['storage']<=EMAX+1e-6)).all())

    def test_frozen_transform_does_not_fit_test_data(self):
        rng=np.random.default_rng(2); rows=[]
        for i in range(35):
            rows.append(dict(index=i,econ=rng.normal(size=33),price=rng.normal(size=144),
                             net=rng.normal(size=144),marginal_low=.8,marginal_high=.2,linear=.5))
        model=ValueModel('quadratic','hybrid',1.,3).fit(rows[:30])
        before=model.pcas['price'].components_.copy(); mean=model.scale.mean_.copy()
        model.predict(rows[30:])
        np.testing.assert_array_equal(before,model.pcas['price'].components_)
        np.testing.assert_array_equal(mean,model.scale.mean_)
        with self.assertRaises(AssertionError): model.predict(rows[:1])


if __name__=='__main__':
    unittest.main()
