import numpy as np

a = 40
g = np.array([1., 1., 1., 1.])

h = a * g
x = np.array([[2.], [2.], [2.], [2.], [2.]])
hs = h.size
xs = x.size
who = np.concatenate((np.ones((hs, xs)), np.zeros((hs, hs))), axis=1)

y_assumed = np.array([10, 14, 3, -13])
error = 100
while error != 0.:
	h = a * g.reshape((hs, 1))
	print(h.shape, x.shape)
	ht = np.concatenate((x, h), axis=0).reshape((xs+hs, 1))
	y = who @ ht
	print(y.ravel())
	print(y_assumed)
	loss = y.reshape((hs, 1)) - y_assumed.reshape((hs, 1))
	error = np.sum(np.abs(loss))

	dy = loss
	dwho = dy @ ht.T
	dx = who[:, :xs].T @ dy
	dh = who[:, xs:].T @ dy
	da = np.sum(dh * g)
	a -= 1e-5 * da
	who -= 1e-5 * dwho

	print(error)
