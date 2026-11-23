"""Child-process entry point for an explicitly bound SkillOpt EnvAdapter."""
from .skillopt_environment import main

if __name__ == "__main__":
    raise SystemExit(main())
