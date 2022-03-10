import lin_q_policy as lq
import numpy as np

# TEST CORRECT BEHAVIOUR FOR SIMPLE CASE
t = lq.LinearQPolicy(np.array([[1,2,3],[4,5,6]]))
feat = np.array([[1,1]])
q = t.regress_q(feat)
print(q)
one_feat = np.array([1,2])
print(t.optimal_action(one_feat))
