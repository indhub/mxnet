<opt_vars(solver)>
<if(solver.momentum)>
momentum = <solver.momentum>
<endif>

optimizer_params={'learning_rate':base_lr<\\>
<if(solver.wd)>, 'wd':wd<endif><\\>
<if(solver.momentum)>, 'momentum':momentum<endif>}<\\>

module.init_optimizer(optimizer='NAG', optimizer_params=optimizer_params)
