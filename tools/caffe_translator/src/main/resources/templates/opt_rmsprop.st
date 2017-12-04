<opt_vars(solver)>
<if(solver.rms_decay)>
gamma1 = <solver.rms_decay>
<endif>
<if(solver.delta)>
epsilon = <solver.delta>
<endif>

optimizer_params={'learning_rate':base_lr<\\>
<if(solver.wd)>, 'wd':wd<endif><\\>
<if(solver.rms_decay)>, 'gamma1':gamma1<endif><\\>
<if(solver.delta)>, 'epsilon':epsilon<endif>}<\\>

module.init_optimizer(optimizer='RMSProp', optimizer_params=optimizer_params)
