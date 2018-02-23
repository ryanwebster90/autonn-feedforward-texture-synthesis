function state = accumulateGradientsAutoNN(net, state, params, batchSize, parserv)
% -------------------------------------------------------------------------

% ensure supported training methods are ordered as expected
assert(isequal(Param.trainMethods, {'gradient', 'average', 'none'})) ;

paramVars = [net.params.var] ;
w = net.getValue(paramVars) ;
dw = net.getDer(paramVars) ;
if isscalar(paramVars), w = {w} ; dw = {dw} ; end

if ~params.plotDiagnostics
  % allow memory to be released, for parameters and their derivatives
  net.setValue([paramVars, paramVars + 1], cell(1, 2 * numel(paramVars))) ;
else
  % free only parameter memory, as we still need the gradients for plotting the diagnostics
  net.setValue(paramVars, cell(size(paramVars))) ;
end

for p=1:numel(net.params)
  if ~isempty(parserv)
    parDer = parserv.pullWithIndex(p) ;
  else
    parDer = dw{p} ;
  end
  
  switch net.params(p).trainMethod
    case 1
      thisDecay = params.weightDecay * net.params(p).weightDecay ;
      thisLR = params.learningRate * net.params(p).learningRate ;
      
      if thisLR>0 || thisDecay>0
        % Normalize gradient and incorporate weight decay.
        parDer = vl_taccum(1/batchSize, parDer, ...
          thisDecay, w{p}) ;
        
        if isempty(params.solver)
          % Default solver is the optimised SGD.
          % Update momentum.
          state.solverState{p} = vl_taccum(...
            params.momentum, state.solverState{p}, ...
            -1, parDer) ;
          
          delta = state.solverState{p} ;
          
          
          % Update parameters.
          w{p} = vl_taccum(1, w{p}, thisLR, delta) ;
          
        else
          % call solver function to update weights
          [w{p}, state.solverState{p}] = ...
            params.solver(w{p}, state.solverState{p}, ...
            parDer, params.solverOpts, thisLR) ;
        end
      end
      
    case 2 % mainly for batch normalization
      thisLR = net.params(p).learningRate ;
      w{p} = vl_taccum(...
        1 - thisLR, w{p}, ...
        (thisLR/batchSize/net.params(p).fanout),  parDer) ;
      
    case 3  % none
    otherwise
      error('Unknown training method ''%i'' for parameter ''%s''.', ...
        net.params(p).trainMethod, ...
        net.params(p).name) ;
  end
  
  
end

if isscalar(paramVars), w = w{1} ; end
net.setValue(paramVars, w) ;