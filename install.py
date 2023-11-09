import launch

if not launch.is_installed("nevergrad"):
    launch.run_pip("install nevergrad==1.0.0", "requirement for quasi random init")
    