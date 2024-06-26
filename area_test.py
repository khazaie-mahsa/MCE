import numpy as np
from scipy.integrate import trapezoid, simpson

# Given points
x = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512], dtype=float)
# y = np.array([0.6601, 0.6593, 0.6576, 0.6541, 0.6479, 0.6353, 0.6219, 0.5032, 0.3133, 0.3133], dtype=float)
# -----------------------------------------------------------------------------
# gt, mask2(bce)
# y = np.array([5.6194,5.4779,5.0293,4.5654,4.7852,4.2969,9.3750,0.,0.,0.],dtype=float)
# Area under the curve (Trapezoidal Rule): 1.299520254403131
# Area under the curve (Simpson's Rule): 0.6628764187866927
# ------------------------------------------------
# gt, mask3 (bce)
# y = np.array([5.5080,5.5450,5.4688,5.2490,4.8828,4.2969,9.3750,0.,0.,0.])
# Area under the curve (Trapezoidal Rule): 1.3125070450097847
# Area under the curve (Simpson's Rule): 0.676429843444227
# ----------------------------------------------------------------------------------
# gt, gt (bce_logits)
# y = np.array([0.6601, 0.6593, 0.6576, 0.6541, 0.6479, 0.6353, 0.6219, 0.5032, 0.3133, 0.3133])
# Normalize the x coordinates between 0 and 1
# ---------------------------------------------------------------------------------
# gt, mask2 (bce +edge detection + simpson)
# miou: 0.4341183238089726
# distances: [tensor(1.4084), tensor(2.3483), tensor(4.2358), tensor(7.2754), tensor(9.6680), tensor(12.5000), tensor(15.6250), tensor(0.), tensor(0.), tensor(0.)]
# mce: 4.731752298113107
#----------------------------------------------------------------------------------
# gt, mask3 (bce +edge detection + simpson)
# miou: 0.507477277506363
# distances: [tensor(1.6388), tensor(2.4628), tensor(4.1199), tensor(7.5195), tensor(9.5703), tensor(10.5469), tensor(9.3750), tensor(0.), tensor(0.), tensor(0.)]
# mce: 3.816401332533446
# ----------------------------------------------------------------------------------
# gt, mask3 (bce witg logits + edge dtection + simpson)
# miou: 0.507477277506363
# distances: [tensor(0.6986), tensor(0.7013), tensor(0.7073), tensor(0.7186), tensor(0.7182), tensor(0.7013), tensor(0.6801), tensor(0.5032), tensor(0.3133), tensor(0.3133)]
# mce: 1.284634014295964
# ------------------------------------------------------------------------------------
# gt, mask (bce with logits + edge detection + simpson)
# miou: 0.4341183238089726
# distances: [tensor(0.6972), tensor(0.7006), tensor(0.7080), tensor(0.7171), tensor(0.7164), tensor(0.7096), tensor(0.7038), tensor(0.5032), tensor(0.3133), tensor(0.3133)]
# mce: 1.288250358923175
# -------------------------------------------------------------------------------
x_normalized = (x - x.min()) / (x.max() - x.min())

# Calculate the area under the curve using the trapezoidal rule
area_trapz = trapezoid(y, x_normalized)
print(f"Area under the curve (Trapezoidal Rule): {area_trapz}")

# Calculate the area under the curve using Simpson's rule
area_simps = simpson(y, x_normalized)
print(f"Area under the curve (Simpson's Rule): {area_simps}")
