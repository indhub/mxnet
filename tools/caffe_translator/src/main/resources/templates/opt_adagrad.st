<opt_vars(solver)>
<if(solver.delta)>
epsilon = <solver.delta>
<endif>

optimizer_params={'learning_rate':base_lr<\\>
<if(solver.wd)>, 'wd':wd<endif><\\>
<if(solver.delta)>, 'epsilon':epsilon<endif>}<\\>

module.init_optimizer(optimizer='AdaGrad', optimizer_params=optimizer_params)
