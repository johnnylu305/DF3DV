# Dataset configuration for training-time evaluation.
#
# This file defines the supported datasets, scene IDs, evaluation image IDs,
# a smaller debug evaluation subset, W&B visualization examples, and the
# method list.


# dataset name -> list of scene IDs
DATASET_IDS = {
    "on-the-go": [
        "corner",
        "fountain",
        "mountain",
        "patio",
        "patio_high",
        "spot",
    ],
    "DF3DV-41-Fixer": [
        "021125-Chess",
        "021125-IslandFactory",
        "021125-OperaHouse",
        "021125-OperaHouse3",
        "021125-TunnelEntrance",
        "031125-Koala",
        "031125-Teapot",
        "031125-TomatoPaste",
        "061125-Actor",
        "061125-Actor2",
        "061125-CarWindow",
        "061125-LightRail",
        "061125-WashCar",
        "071125-BoardGame",
        "071125-CD",
        "071125-Cut",
        "071125-GlassyContainers",
        "071125-MusicBox",
        "071125-Yarn",
        "081125-RollingShutter",
        "081125-RollingShutter2",
        "101125-Light",
        "101125-Shadow",
        "101125-Shadow2",
        "101125-Train",
        "101125-Water",
        "111125-Flower",
        "111125-Light",
        "111125-Straw",
        "131125-AluminiumFoil",
        "131125-BlackBackground",
        "131125-BlackTable",
        "131125-Head",
        "131125-TrinketBox",
        "191125-Cars",
        "191125-NightOperaHouse",
        "191125-NightOperaHouse2",
        "301025-Statue",
        "301025-Temple",
        "301025-TempleBellIncense",
        "301025-TempleDrumIncense",
    ],
}

# four evaluation images per scene
EVAL_IDS = [
    # DF3DV-41-Fixer
    "021125-Chess_extra_IMG_7520", "021125-Chess_extra_IMG_7480", "021125-Chess_extra_IMG_7494", "021125-Chess_extra_IMG_7499",
    "021125-IslandFactory_extra_IMG_7727", "021125-IslandFactory_extra_IMG_7744", "021125-IslandFactory_extra_IMG_7757", "021125-IslandFactory_extra_IMG_7759",
    "021125-OperaHouse_extra_IMG_7158", "021125-OperaHouse_extra_IMG_7162", "021125-OperaHouse_extra_IMG_7163", "021125-OperaHouse_extra_IMG_7167",
    "021125-OperaHouse3_extra_IMG_7394", "021125-OperaHouse3_extra_IMG_7396", "021125-OperaHouse3_extra_IMG_7402", "021125-OperaHouse3_extra_IMG_7408",
    "021125-TunnelEntrance_extra_IMG_7835", "021125-TunnelEntrance_extra_IMG_7838", "021125-TunnelEntrance_extra_IMG_7840", "021125-TunnelEntrance_extra_IMG_7842",
    "031125-Koala_extra_IMG_8133", "031125-Koala_extra_IMG_8137", "031125-Koala_extra_IMG_8143", "031125-Koala_extra_IMG_8153",
    "031125-Teapot_extra_IMG_8027", "031125-Teapot_extra_IMG_8032", "031125-Teapot_extra_IMG_8037", "031125-Teapot_extra_IMG_8042",
    "031125-TomatoPaste_extra_IMG_7926", "031125-TomatoPaste_extra_IMG_7929", "031125-TomatoPaste_extra_IMG_7931", "031125-TomatoPaste_extra_IMG_7947",
    "061125-Actor_extra_IMG_8291", "061125-Actor_extra_IMG_8321", "061125-Actor_extra_IMG_8324", "061125-Actor_extra_IMG_8326",
    "061125-Actor2_extra_IMG_8334", "061125-Actor2_extra_IMG_8341", "061125-Actor2_extra_IMG_8345", "061125-Actor2_extra_IMG_8347",
    "061125-CarWindow_extra_IMG_8716", "061125-CarWindow_extra_IMG_8722", "061125-CarWindow_extra_IMG_8723", "061125-CarWindow_extra_IMG_8730",
    "061125-LightRail_extra_IMG_8431", "061125-LightRail_extra_IMG_8439", "061125-LightRail_extra_IMG_8442", "061125-LightRail_extra_IMG_8467",
    "061125-WashCar_extra_IMG_8584", "061125-WashCar_extra_IMG_8590", "061125-WashCar_extra_IMG_8596", "061125-WashCar_extra_IMG_8581",
    "071125-BoardGame_extra_IMG_9372", "071125-BoardGame_extra_IMG_9374", "071125-BoardGame_extra_IMG_9375", "071125-BoardGame_extra_IMG_9398",
    "071125-CD_extra_IMG_9209", "071125-CD_extra_IMG_9211", "071125-CD_extra_IMG_9240", "071125-CD_extra_IMG_9204",
    "071125-Cut_extra_IMG_8929", "071125-Cut_extra_IMG_8953", "071125-Cut_extra_IMG_8957", "071125-Cut_extra_IMG_8968",
    "071125-GlassyContainers_extra_IMG_9081", "071125-GlassyContainers_extra_IMG_9087", "071125-GlassyContainers_extra_IMG_9096", "071125-GlassyContainers_extra_IMG_9111",
    "071125-MusicBox_extra_IMG_9451", "071125-MusicBox_extra_IMG_9456", "071125-MusicBox_extra_IMG_9457", "071125-MusicBox_extra_IMG_9460",
    "071125-Yarn_extra_IMG_9298", "071125-Yarn_extra_IMG_9302", "071125-Yarn_extra_IMG_9310", "071125-Yarn_extra_IMG_9318",
    "081125-RollingShutter_extra_IMG_1867", "081125-RollingShutter_extra_IMG_1875", "081125-RollingShutter_extra_IMG_1877", "081125-RollingShutter_extra_IMG_1886",
    "081125-RollingShutter2_extra_IMG_2009", "081125-RollingShutter2_extra_IMG_2016", "081125-RollingShutter2_extra_IMG_2022", "081125-RollingShutter2_extra_IMG_2034",
    "101125-Light_extra_IMG_9960", "101125-Light_extra_IMG_9967", "101125-Light_extra_IMG_9975", "101125-Light_extra_IMG_9980",
    "101125-Shadow_extra_IMG_9857", "101125-Shadow_extra_IMG_9864", "101125-Shadow_extra_IMG_9870", "101125-Shadow_extra_IMG_9883",
    "101125-Shadow2_extra_IMG_9678", "101125-Shadow2_extra_IMG_9681", "101125-Shadow2_extra_IMG_9685", "101125-Shadow2_extra_IMG_9711",
    "101125-Train_extra_IMG_9536", "101125-Train_extra_IMG_9592", "101125-Train_extra_IMG_9559", "101125-Train_extra_IMG_9597",
    "101125-Water_extra_IMG_9761", "101125-Water_extra_IMG_9762", "101125-Water_extra_IMG_9775", "101125-Water_extra_IMG_9778",
    "111125-Flower_extra_IMG_0160", "111125-Flower_extra_IMG_0169", "111125-Flower_extra_IMG_0184", "111125-Flower_extra_IMG_0188",
    "111125-Light_extra_IMG_0399", "111125-Light_extra_IMG_0408", "111125-Light_extra_IMG_0413", "111125-Light_extra_IMG_0416",
    "111125-Straw_extra_IMG_0030", "111125-Straw_extra_IMG_0034", "111125-Straw_extra_IMG_0041", "111125-Straw_extra_IMG_0051",
    "131125-AluminiumFoil_extra_IMG_1036", "131125-AluminiumFoil_extra_IMG_1041", "131125-AluminiumFoil_extra_IMG_1050", "131125-AluminiumFoil_extra_IMG_1054",
    "131125-BlackBackground_extra_IMG_0950", "131125-BlackBackground_extra_IMG_0954", "131125-BlackBackground_extra_IMG_0964", "131125-BlackBackground_extra_IMG_0965",
    "131125-BlackTable_extra_IMG_0834", "131125-BlackTable_extra_IMG_0836", "131125-BlackTable_extra_IMG_0839", "131125-BlackTable_extra_IMG_0863",
    "131125-Head_extra_IMG_0695", "131125-Head_extra_IMG_0704", "131125-Head_extra_IMG_0721", "131125-Head_extra_IMG_0731",
    "131125-TrinketBox_extra_IMG_0740", "131125-TrinketBox_extra_IMG_0746", "131125-TrinketBox_extra_IMG_0751", "131125-TrinketBox_extra_IMG_0753",
    "191125-Cars_extra_20251119_205815", "191125-Cars_extra_20251119_205928", "191125-Cars_extra_20251119_205728", "191125-Cars_extra_20251119_210031",
    "191125-NightOperaHouse_extra_20251119_204656", "191125-NightOperaHouse_extra_20251119_204735", "191125-NightOperaHouse_extra_20251119_204812", "191125-NightOperaHouse_extra_20251119_204901",
    "191125-NightOperaHouse2_extra_20251119_203021", "191125-NightOperaHouse2_extra_20251119_203113", "191125-NightOperaHouse2_extra_20251119_203203", "191125-NightOperaHouse2_extra_20251119_203228",
    "301025-Statue_extra_IMG_6691", "301025-Statue_extra_IMG_6700", "301025-Statue_extra_IMG_6721", "301025-Statue_extra_IMG_6733",
    "301025-Temple_extra_IMG_6383", "301025-Temple_extra_IMG_6463", "301025-Temple_extra_IMG_6380", "301025-Temple_extra_IMG_6468",
    "301025-TempleBellIncense_extra_IMG_5838", "301025-TempleBellIncense_extra_IMG_5839", "301025-TempleBellIncense_extra_IMG_5841", "301025-TempleBellIncense_extra_IMG_5846",
    "301025-TempleDrumIncense_extra_IMG_6104", "301025-TempleDrumIncense_extra_IMG_6106", "301025-TempleDrumIncense_extra_IMG_6114", "301025-TempleDrumIncense_extra_IMG_6134",
    # on-the-go
    "corner_1extra000", "corner_1extra005", "corner_1extra006", "corner_1extra011",
    "fountain_1extra001", "fountain_1extra002", "fountain_1extra007", "fountain_1extra000",
    "mountain_1extra117", "mountain_1extra120", "mountain_1extra124", "mountain_1extra128",
    "patio_1extra079", "patio_1extra088", "patio_1extra100", "patio_1extra087",
    "patio_high_1extra002", "patio_high_1extra008", "patio_high_1extra025", "patio_high_1extra026",
    "spot_1extra168", "spot_1extra171", "spot_1extra174", "spot_1extra175",
]

# small evaluation set for debugging and faster validation
EVAL_SMALL_IDS = [
    # DF3DV-41-Fixer
    "021125-Chess_extra_IMG_7520",
    "021125-Chess_extra_IMG_7480",
    "021125-Chess_extra_IMG_7494",
    "021125-Chess_extra_IMG_7499",
    # on-the-go
    "patio_high_1extra002",
    "patio_high_1extra008",
    "patio_high_1extra025",
    "patio_high_1extra026",
]

# images used for W&B visualization
WANDB_IDS = [
    # DF3DV-41-Fixer
    "021125-Chess_extra_IMG_7520",
    "021125-OperaHouse3_extra_IMG_7394",
    "031125-TomatoPaste_extra_IMG_7926",
    "061125-Actor2_extra_IMG_8334",
    "071125-MusicBox_extra_IMG_9451",
    "081125-RollingShutter2_extra_IMG_2009",
    "101125-Shadow_extra_IMG_9857",
    "131125-AluminiumFoil_extra_IMG_1036",
    "131125-BlackTable_extra_IMG_0834",
    "131125-Head_extra_IMG_0695",
    # on-the-go
    "patio_high_1extra002",
]

# method list
METHODS = [
    "T3DGS_TMR",
    "SLS",
    "RBSPLAT",
    "WILDGS",
    "ASYMGS",
    "DEGS",
    "T3DGS",
    "OCSPLAT",
    "DESPLAT",
    "3DGS",
]
