import near.dsl as dsl

DSL_DICT = {
    ('list', 'list') : [dsl.MapFunction, dsl.MapPrefixesFunction, dsl.SimpleITE],
    ('list', 'atom') : [dsl.MapAverageFunction,dsl.SimpleITE],
    ('atom', 'atom') : [dsl.SimpleITE,dsl.AddFunction, dsl.MultiplyFunction, 
                        dsl.ResMARSHeadBodyAngleComputation, dsl.IntrMARSHeadBodyAngleComputation, 
                        dsl.MARSNoseNoseDistanceComputation,dsl.MARSNoseTailDistanceComputation,
                        dsl.ResMARSSocialAngleComputation, dsl.IntrMARSSocialAngleComputation, 
                        dsl.ResMARSSpeedComputation, dsl.IntrMARSSpeedComputation]
}


CUSTOM_EDGE_COSTS = {
    ('list', 'list') : {},
    ('list', 'atom') : {},
    ('atom', 'atom') : {}
}
