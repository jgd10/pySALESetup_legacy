import random
import warnings
import numpy as np
from PIL import Image
from math import ceil
import cPickle as pickle
import scipy.special as scsp
from scipy import stats as scst
import matplotlib.pyplot as plt
import matplotlib.path   as mpath
from collections import Counter, OrderedDict
from mpl_toolkits.axes_grid1 import make_axes_locatable

from functions import *
from grainclasses import *
from objectclasses import *
from domainclasses import *

print " ===================================================== "
print "               _______   __   ________    __           "
print "    ___  __ __/ __/ _ | / /  / __/ __/__ / /___ _____  "
print "   / _ \/ // /\ \/ __ |/ /__/ _/_\ \/ -_) __/ // / _ \ "
print "  / .__/\_, /___/_/ |_/____/___/___/\__/\__/\_,_/ .__/ "
print " /_/   /___/                                   /_/     "
print "                                      by J. G. Derrick "
print " ===================================================== "
