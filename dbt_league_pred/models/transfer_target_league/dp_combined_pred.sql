With binary_pred as (
    Select
        *
    From {{ref('dp_binary_pred')}}
),
multi_class as (
    Select
        *
    From {{ref('dp_multiclass_pred')}}
)
Select
    bp.player_id,
    bp.player,
    bp.top_league,
    bp.non_top_league,
    coalesce(mc.bundesliga,0.0) as bundesliga,
    coalesce(mc.laliga,0.0) as laliga,
    coalesce(mc.ligue_1,0.0) as ligue_1,
    coalesce(mc.premier_league,0.0) as premier_league,
    coalesce(mc.serie_a,0.0) as serie_a,
    if(mc.creation_timestamp is null, bp.creation_timestamp,mc.creation_timestamp) as creation_timestamp
From binary_pred bp
left join multi_class mc
    on bp.Player_ID = mc.Player_ID