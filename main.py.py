import numpy as np
import matplotlib.pyplot as pl
import matplotlib.animation as anim
import seaborn as sns

sns.set(rc={"figure.figsize": (14, 9)})
sns.set_style("darkgrid")
curPal = ['#3385ff', '#00b300', '#a64dff']
sns.set_palette(curPal, n_colors=3)

def sigmoid(t):

	t_trans = 10.0

	k = 10.0/t_trans

	return np.reciprocal(1 + np.exp(-k*t))

def sigmoid_grad(t, t_on):
	
    fx = sigmoid(t - t_on)
    
    return fx*(1 - fx)

def SLS(t, g0, g1, tau):
	return g0 + g1*np.exp(-t/tau)
	
############
# data get #
############

# change resolution to change apparent speed of animation
t = np.linspace(0.0, 650.0, 6500)
#~ t = np.linspace(0.0, 650.0, 3250)
#~ t = np.linspace(0.0, 650.0, 800)

length = len(t)

sls_t = SLS(t, 0.15, 0.5, 100.0)
sls_t_padded = np.pad(sls_t, (len(sls_t), len(sls_t)), 'constant', constant_values=(0, 0))
sls_t_padded = np.flipud(sls_t_padded) # we want convolution NOT correlation!

de_dt = sigmoid_grad(t, 150.0) - sigmoid_grad(t, 500.0)

convolved = np.multiply(np.convolve(sls_t, de_dt, mode='full')[0:len(t)], np.gradient(t))

############
# plotting #
############

fig, ax = pl.subplots()

ax.set_ylim(-0.55, 0.75)

de_line, = ax.plot(t, de_dt)
sls_line, = ax.plot(t[0], sls_t_padded[2*length], '--')
convolved_line, = ax.plot(t[0], convolved[0]) 

lines = [de_line, sls_line, convolved_line]

def init():
	
	lines[0].set_data(t, de_dt)
	lines[1].set_data(t[0], sls_t_padded[2*length])
	lines[2].set_data(t[0], convolved[0])
	    
	return lines

def animate(i):
	
	if i<length:
		lines[1].set_data(t[0:i], sls_t_padded[(2*length - i):(2*length)])
		lines[2].set_data(t[0:i], convolved[0:i])
	else:
		# in you want SLS to keep sliding
		lines[1].set_data(t[(i - length):i], sls_t_padded[length:(3*length - i)])
	
	return lines
	
ani = anim.FuncAnimation(fig, animate, np.arange(1, length), interval=1, init_func=init, blit=True)

pl.show()


