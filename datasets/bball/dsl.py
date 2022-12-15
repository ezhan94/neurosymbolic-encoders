import near.dsl as dsl


DSL_DICT = {
    ('list', 'atom') : [dsl.MapAverageFunction],
    ('atom', 'atom') : [dsl.BBallSpeed, dsl.BBallXPos, dsl.BBallYPos, dsl.BBallDist2Basket,
                        dsl.BBallPlayerAvg, dsl.BBallPlayerMax, dsl.BBallPlayerMin]
}

CUSTOM_EDGE_COSTS = {
    ('list', 'list') : {},
    ('list', 'atom') : {},
    ('atom', 'atom') : {}
}
