import near.dsl as dsl


DSL_DICT = {
    ('list', 'atom') : [dsl.FinalXPosition, dsl.FinalYPosition]
    # ('list', 'atom') : [dsl.FinalXPosition, dsl.FinalYPosition, dsl.AvgSpeed, dsl.AvgAccel]
}

CUSTOM_EDGE_COSTS = {
    ('list', 'list') : {},
    ('list', 'atom') : {},
    ('atom', 'atom') : {}
}
