from . import algorithms, common

from intrl.common.safety.geoms import BoundarySigwalls

from safety_gymnasium.assets import geoms 
geoms.BoundarySigwalls = BoundarySigwalls
geoms.GEOMS_REGISTER += [BoundarySigwalls]
