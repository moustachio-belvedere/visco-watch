import numpy as np
import matplotlib.pyplot as pl
import matplotlib.animation as anim
import seaborn as sns

sns.set(rc={"figure.figsize": (14, 9)})
sns.set_style("darkgrid")

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
#~ t = np.linspace(0.0, 1000.0, 4000)
t = np.linspace(0.0, 1000.0, 2000)
#~ t = np.linspace(0.0, 1000.0, 1000)

length = len(t)

sls_t = SLS(t, 0.15, 0.5, 100.0)
sls_t_padded = np.pad(sls_t, (len(sls_t), len(sls_t)), 'constant', constant_values=(0, 0))
sls_t_padded = np.flipud(sls_t_padded) # we want convolution NOT correlation!

e = sigmoid(t - 150.0) - sigmoid(t - 500.0)
de_dt = sigmoid_grad(t, 150.0) - sigmoid_grad(t, 500.0)

convolved = np.multiply(np.convolve(sls_t, de_dt, mode='full')[0:len(t)], np.gradient(t))

############
# plotting #
############

# global set up
grepeat = True
	
fig, axarr = pl.subplots(2, 2)

# axarr[0,0] set up
e_line, = axarr[0,0].plot(t, e, "-", label="strain load", color="#566573")
de_line00, = axarr[0,0].plot(t, de_dt, "--", label="d(strain)/dt", color="#3385ff")
dotfollow_line, = axarr[0,0].plot(t[0], e[0], "o", color="#dc7633") 

axarr[0,0].legend(loc='lower left')

axarr[0,0].set_xlim(0, 1000)
axarr[0,0].set_ylim(-0.4, 1.2)

lines00 = (e_line, de_line00, dotfollow_line)

def init00():
	
	lines00[0].set_data(t, e)
	lines00[1].set_data(t, de_dt)
		
	return lines00
	
def animate00(i):
	
	lines00[2].set_data(t[i], e[i]) 
	
	return lines00

ani00 = anim.FuncAnimation(fig, animate00, np.arange(1, length-1), interval=1, init_func=init00, blit=True, repeat=grepeat)
	
# axarr[1,0] set up
axarr[1,0].plot(t, sls_t, label="G_SLS(t)", color="#00b300")
axarr[1,0].legend(loc='lower left')

axarr[1,0].set_xlim(0, 1000)
axarr[1,0].set_ylim(0.0, 0.75)

# axarr[0,1] set up
axarr[0,1].set_xlim(0, 1000)
axarr[0,1].set_ylim(-0.55, 0.75)

de_line, = axarr[0,1].plot(t, de_dt, label="d(strain)/dt", color="#3385ff")
sls_line, = axarr[0,1].plot(t, sls_t_padded[(2*length - 1):(3*length - 1)], '-.', label="0 padded G_SLS(-t)", color="#00b300")

lines01 = (de_line, sls_line)

def init01():
	
	lines01[0].set_data(t, de_dt)
	    
	return lines01

def animate01(i):
	
	lines01[1].set_ydata(sls_t_padded[(2*length - i):(3*length - i)])
	
	return lines01

axarr[0,1].legend(loc='lower left')
ani01 = anim.FuncAnimation(fig, animate01, np.arange(1, length-1), interval=1, init_func=init01, blit=True, repeat=grepeat)

# axarr[1,1] set up
axarr[1,1].set_xlim(0, 1000)
axarr[1,1].set_ylim(-0.55, 0.75)

multiplied_scaling = 2.0

convolved_line, = axarr[1,1].plot(t[0], convolved[0], "-.", label="convolution", color="#dc7633")
multiplied_line, = axarr[1,1].plot(t[0], multiplied_scaling*np.multiply(de_dt, sls_t_padded[(2*length - 1):(3*length - 1)])[0], "-", label="multiplication", color="#a64dff", alpha=0.7)

lines11 = (convolved_line, multiplied_line)

def init11():
	
	return lines11
	
def animate11(i):
	
	lines11[0].set_data(t[0:i], convolved[0:i])
	#~ lines11[1].set_data(t[0:i], multiplied_scaling*np.multiply(de_dt, sls_t_padded[(2*length - i):(3*length - i)])[0:i])
	lines11[1].set_data(t, multiplied_scaling*np.multiply(de_dt, sls_t_padded[(2*length - i):(3*length - i)]))
	
	return lines11

axarr[1,1].legend(loc='lower left')
ani11 = anim.FuncAnimation(fig, animate11, np.arange(1, length-1), interval=1, init_func=init11, blit=True, repeat=grepeat)

pl.show()


