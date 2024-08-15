from signal_system import SignalSystem

cfg = 'config/optimize.ini'
signal_system = SignalSystem(cfg)

signal_system.train_detect_module()