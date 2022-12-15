# Default DSL
from .neural_functions import HeuristicNeuralFunction, ListToListModule, ListToAtomModule, AtomToAtomModule, init_neural_function
from .library_functions import StartFunction, LibraryFunction, \
                                MapFunction, MapAverageFunction, MapPrefixesFunction, \
                                ITE, SimpleITE, \
                                FoldFunction, FullInputAffineFunction, AddFunction, MultiplyFunction

# Additional running average functions
from .running_averages import RunningAverageFunction, RunningAverageLast5Function, RunningAverageLast10Function, \
                                RunningAverageWindow7Function, RunningAverageWindow5Function, RunningAverageWindow11Function

# Domain-specific library functions
from .crim13 import Crim13PositionSelection, Crim13DistanceSelection, Crim13DistanceChangeSelection, \
                    Crim13VelocitySelection, Crim13AccelerationSelection, Crim13AngleSelection, Crim13AngleChangeSelection
from .fruitflies import FruitFlyWingSelection, FruitFlyRatioSelection, FruitFlyPositionalSelection, \
                        FruitFlyAngularSelection, FruitFlyLinearSelection
from .basketball import BBallSpeed, BBallXPos, BBallYPos, BBallDist2Basket, \
                        BBallPlayerAvg, BBallPlayerMax, BBallPlayerMin
from .synthetic import FinalXPosition, FinalYPosition, AvgSpeed, AvgAccel
from .mars import ResMARSHeadBodyAngleComputation, IntrMARSHeadBodyAngleComputation, \
                    MARSNoseNoseDistanceComputation, MARSNoseTailDistanceComputation, ResMARSSocialAngleComputation, \
                    IntrMARSSocialAngleComputation, ResMARSSpeedComputation, IntrMARSSpeedComputation


# For importing DSLs and program graph edge costs
def import_dsl(module_name):
    import importlib
    dsl_module = importlib.import_module(module_name)
    return dsl_module.DSL_DICT, dsl_module.CUSTOM_EDGE_COSTS
