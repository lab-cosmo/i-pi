from ipi.inputs.simulation import InputSimulation
from ipi.utils.io.inputs import io_xml
import numpy as np

path2iipi="./input.xml"

ifile = open(path2iipi, "r")
xmlrestart = io_xml.xml_parse_file(ifile)
ifile.close()
isimul = InputSimulation()
isimul.parse(xmlrestart.fields[0][1])
simul = isimul.fetch()


prefix =  simul.syslist[0].motion.prefix
max_iter = simul.syslist[0].motion.max_iter
m3 = simul.syslist[0].beads.m3[-1]
M = np.diag(m3)
iM = np.diag(1.0 / m3)
sqM = np.diag(np.sqrt(m3))
isqM = np.diag(1.0 / np.sqrt(m3))
kbT = float(simul.syslist[0].ensemble.temp)
beta = 1.0 / kbT
findex = max_iter - 1
findex = 18
widening = 1.10

for i in range(max_iter):
    # Imports the q, iD, x, f from the i^th  SCP iteration.
    iD0 = np.loadtxt(prefix + ".iD." + str(i))
    q0 = np.loadtxt(prefix + ".q." + str(i))
    K0 = np.loadtxt(prefix + ".K." + str(i))
    V0 = np.loadtxt(prefix + ".V0." + str(i))
    # Calculates the frequencies of the trial Hamiltonian.
    hw0 = np.loadtxt(prefix + ".w." + str(i))[3:]
    betahw0 = beta * hw0
    vH0 = V0 + np.sum(hw0 * np.cosh(betahw0 / 2.0) / np.sinh(betahw0 / 2.0) * 0.250)
    AH0 = V0 + np.sum(hw0 * 0.5 + kbT * np.log(1 - np.exp(-betahw0)))
    adv = 0.0
    vdv = 0.0
    norm = 0

    # Inner loop over previous SCP steps
    for j in range(i + 1):
    
      # Imports the q, iD, x, f from the i^th  SCP iteration.
      iD = np.loadtxt(prefix + ".iD." + str(j)) / widening**2
      q = np.loadtxt(prefix + ".q." + str(j))
      K = np.loadtxt(prefix + ".K." + str(j))
      V = np.loadtxt(prefix + ".V0." + str(j))

      x = np.loadtxt(prefix + ".x." + str(j))
      v = np.loadtxt(prefix + ".v." + str(j))
      vh = V0 + 0.5 * np.sum(np.dot(x - q0, K0.T) * (x - q0), axis=1)      

      # Calculates the weights to be used for reweighting.
      w = np.exp(-(0.50 * np.dot(iD0, (x - q0).T).T * (x - q0)).sum(axis=1) + (0.50 * np.dot(iD, (x - q).T).T * (x - q)).sum(axis=1))
      V1 = np.sum(w)
      V2 = np.sum(w**2)
      advj = np.sum(w * (v - vh)) / V1
      vdvj = np.sum(w * (v - vh - advj)**2) / (V1 - V2 / V1)
      c = 1.0 / vdvj
      norm += c
      adv += c * advj
      vdv += c**2 * vdvj
    adv = adv / norm
    vdv = vdv / norm**2 / len(w)

    print "F :", i, AH0, AH0 + adv, np.sqrt(vdv)
