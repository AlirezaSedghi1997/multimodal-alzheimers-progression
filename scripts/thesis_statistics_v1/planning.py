"""Retrospective Riley scenarios, including the published null-likelihood correction."""
import math

def riley_scenarios(rate,mean_followup):
    if rate<=0 or mean_followup<=0:raise ValueError('POSITIVE_RATE_AND_FOLLOWUP_REQUIRED')
    max_r2=1-math.exp(2*rate*mean_followup*(math.log(rate)-1))
    if not 0<max_r2<1:raise ValueError('INVALID_COX_SNELL_UPPER_BOUND')
    horizon,margin,z=3.0,0.05,1.96
    survival=math.exp(-rate*horizon)
    delta=math.log1p(margin/survival)
    n_precision=math.ceil(rate*(z*horizon/delta)**2/mean_followup)
    rows=[]
    for parameters in [7,8,12]:
        for r2 in [.05,.10,.15,.20]:
            shrinkage=.9
            n1=math.ceil(parameters/((shrinkage-1)*math.log(1-r2/shrinkage)))
            s2=r2/(r2+.05*max_r2)
            n2=math.ceil(parameters/((s2-1)*math.log(1-r2/s2)))
            n=max(n1,n2,n_precision)
            se=math.sqrt(rate/(n*mean_followup))
            lower=1-math.exp(-(rate-z*se)*horizon)
            upper=1-math.exp(-(rate+z*se)*horizon)
            risk=1-survival
            if max(risk-lower,upper-risk)>margin:raise ValueError('RISK_PRECISION_NOT_MET')
            rows.append(dict(parameters=parameters,assumed_cox_snell_r2=r2,n_shrinkage=n1,
                n_optimism=n2,n_risk_precision=n_precision,minimum_for_scenario=n,risk_interval=[lower,upper]))
    return dict(scope='RETROSPECTIVE_SCENARIOS_NOT_PROSPECTIVE_DESIGN_NOT_CNN_ADEQUACY',
        assumptions=dict(rate_per_person_year=rate,mean_observed_years=mean_followup,
            constant_hazard_for_planning_only=True,risk_horizon_years=horizon,
            absolute_risk_margin=margin,maximum_cox_snell_r2=max_r2,expected_r2_is_assumed_not_fitted=True),
        rows=rows,references=['https://doi.org/10.1002/sim.7992','https://doi.org/10.1002/sim.8409'])
