from __future__ import annotations


# Human review uses this stable vocabulary before allowing a one-off custom tag.
# The semantic model can still propose a more specific value, but reviewers are
# deliberately steered toward these reusable library shelves first.
PRIMARY_FAMILIES = [
    "animal",
    "atmosphere",
    "bird",
    "blood",
    "cloud",
    "creature",
    "crowd",
    "debris",
    "distortion",
    "dust",
    "electricity",
    "energy",
    "environment",
    "explosion",
    "fire",
    "fluid",
    "fog",
    "light",
    "lens_effect",
    "magic",
    "muzzle_flash",
    "particle",
    "plate",
    "projectile",
    "rain",
    "smoke",
    "snow",
    "spark",
    "steam",
    "trail",
    "vegetation",
    "water",
    "weather",
    "other",
    "unknown",
]


EFFECT_TYPES_BY_FAMILY: dict[str, list[str]] = {
    "animal": ["isolated_animal", "animal_group", "running_animal", "flying_animal"],
    "atmosphere": ["atmospheric_haze", "atmospheric_mist", "wispy_snow", "billowing_snow", "heat_distortion", "atmospheric_dust"],
    "bird": ["flocking_birds", "circling_birds", "flying_birds", "landing_birds", "isolated_bird"],
    "blood": ["blood_hit", "blood_spray", "blood_mist", "blood_drip", "blood_pool"],
    "cloud": ["expanding_cloud", "drifting_cloud", "cloud_tank", "storm_cloud", "low_cloud"],
    "creature": ["creature_performance", "creature_run", "creature_flight", "creature_crowd"],
    "crowd": ["crowd_plate", "running_crowd", "walking_crowd", "crowd_reaction"],
    "debris": ["falling_debris_impact", "debris_burst", "ground_debris", "drifting_debris", "shattered_debris"],
    "distortion": ["heat_distortion", "shockwave_distortion", "energy_distortion", "lens_distortion"],
    "dust": ["ground_dust_hit", "dust_burst", "dust_plume", "falling_dust", "drifting_dust", "dust_trail"],
    "electricity": ["branching_lightning_strike", "lightning_burst", "electrical_arc", "electric_sparks"],
    "energy": ["expanding_energy_burst", "localized_energy_impact", "magical_impact", "energy_beam", "energy_orb", "energy_trail", "shockwave"],
    "environment": ["sky_plate", "landscape_plate", "city_plate", "street_plate", "interior_plate"],
    "explosion": ["fireball_expansion", "side_burst_explosion", "nuclear_blast", "ground_explosion", "aerial_explosion", "explosion_with_debris", "smoke_explosion"],
    "fire": ["ground_fire", "fireball", "fireball_expansion", "fireball_streak", "flame_jet", "torch_flame", "embers", "burning_debris"],
    "fluid": ["ink_tank", "fluid_splash", "fluid_pour", "fluid_simulation", "viscous_fluid"],
    "fog": ["ground_fog", "drifting_fog", "fog_bank", "rising_fog"],
    "light": ["light_ray", "god_rays", "light_sweep", "glow", "strobe"],
    "lens_effect": ["lens_flare", "light_leak", "glint", "bokeh", "lens_dirt", "lens_scratch"],
    "magic": ["magic_dust", "magic_burst", "spell_hit", "spell_trail", "magic_circle"],
    "muzzle_flash": ["front_muzzle_flash", "side_muzzle_flash", "automatic_muzzle_flash", "muzzle_smoke"],
    "particle": ["drifting_particles", "expanding_particle_burst", "sparse_particle_drift", "rising_magic_dust", "falling_particles", "metallic_debris_drifting"],
    "plate": ["greenscreen_performance_plate", "bluescreen_performance_plate", "greenscreen_object_plate", "cg_object_plate", "clean_plate", "environment_plate", "action_plate", "crowd_plate"],
    "projectile": ["projectile_streak", "tracer", "energy_projectile", "projectile_impact"],
    "rain": ["falling_rain", "continuous_rain", "heavy_rain", "light_rain", "distant_rain"],
    "smoke": ["rising_smoke_plume", "smoke_burst", "smoke_trail", "ground_smoke", "drifting_smoke", "dust_hit"],
    "snow": ["falling_snow", "blowing_snow", "snow_flurry", "snow_burst"],
    "spark": ["spark_burst", "spark_shower", "welding_sparks", "electrical_sparks", "embers"],
    "steam": ["rising_steam_plume", "rising_steam_jet", "steam_burst", "drifting_steam"],
    "trail": ["smoke_trail", "fire_trail", "energy_trail", "dust_trail", "projectile_trail"],
    "vegetation": ["falling_leaves", "blowing_leaves", "grass_plate", "tree_plate", "vegetation_debris"],
    "water": ["dripping_water_stream", "falling_drop_impact", "water_splash", "water_spray", "waterfall", "water_mist", "wave"],
    "weather": ["storm", "wind", "hail", "blizzard", "weather_plate"],
    "other": [],
    "unknown": [],
}


FIELD_VOCABULARY: dict[str, list[str]] = {
    "primary_family": PRIMARY_FAMILIES,
    "secondary_families": PRIMARY_FAMILIES,
    "composition.viewpoint": ["front", "side", "top_down", "low_angle", "oblique", "unknown"],
    "composition.shot_scale": ["macro", "close", "medium", "wide", "very_wide", "unknown"],
    "composition.edge_contact": ["top", "bottom", "left", "right", "none", "unknown"],
    "composition.frame_coverage": ["small", "medium", "large", "full_frame", "unknown"],
    "composition.spatial_distribution": ["central", "off_centre", "grounded", "overhead", "scattered", "full_frame", "unknown"],
    "motion.temporal_arc": ["instantaneous", "build_peak_decay", "already_active_decay", "continuous", "burst_then_dissipate", "looped", "static", "other", "unknown"],
    "motion.onset": ["instant", "fast", "gradual", "already_active", "unknown"],
    "motion.speed": ["still", "slow", "medium", "fast", "very_fast", "mixed", "unknown"],
    "motion.character": ["rising", "falling", "expanding", "contracting", "drifting", "circling", "orbiting", "flocking", "flickering", "branching", "turbulent", "impact", "walking", "turning", "gesturing", "posing", "speaking", "interaction", "other"],
    "motion.depth_motion": ["toward_camera", "away_from_camera", "in_plane", "mixed", "unknown"],
    "motion.event_count": ["single", "multiple", "continuous_many", "unknown"],
    "motion.expansion": ["none", "slight", "strong", "contracting", "mixed", "unknown"],
    "motion.loopability": ["clean_loop", "possible_loop", "one_shot", "not_loopable", "unknown"],
    "appearance.backing": ["transparent", "black", "white", "green_screen", "blue_screen", "checkerboard", "environment", "unknown"],
    "appearance.colour": ["black", "dark_grey", "grey", "white", "red", "orange", "yellow", "green", "blue", "purple", "brown", "mixed", "unknown"],
    "appearance.density": ["wispy", "light", "medium", "dense", "opaque", "mixed", "unknown"],
    "appearance.texture": ["fine", "billowing", "chunky", "filament", "turbulent", "smooth", "other"],
    "appearance.lighting": ["self_luminous", "top_lit", "bottom_lit", "side_lit", "front_lit", "backlit", "ambient", "mixed", "unknown"],
}


def vocabulary_for(path: str, primary_family: str | None = None) -> list[str]:
    if path == "effect_type":
        return list(EFFECT_TYPES_BY_FAMILY.get(str(primary_family or "unknown"), []))
    return list(FIELD_VOCABULARY.get(path, []))
