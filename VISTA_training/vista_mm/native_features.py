"""Schema and recipe identities for the prepared native sensor cache."""

NATIVE_SCHEMA = "vista-sensors-native-v2"
WINDOWS = 8
SECONDS = 5.0
RECIPE = {
    "version": 1,
    "windows": WINDOWS,
    "seconds": SECONDS,
    "margin_seconds": 0.025,
    "audio_sample_rate": 16000,
    "microphones": 7,
    "fbank_frames": 498,
    "mel_bins": 128,
    "fbank_mean": 15.41663,
    "fbank_std": 6.55582,
    "imu_sample_rate": 50,
    "imu_samples": 250,
    "depth_grid": 16,
    "incomplete_windows": "masked",
    "microphone_pooling": "mean_after_encoder",
    "imu_units": "native_acceleration_m_s2_angular_velocity_rad_s",
}
