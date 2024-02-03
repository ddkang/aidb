from setuptools import find_packages
import os


def read_requirements():
    requirements = {"main": []}
    current_section = "main"

    with open("requirements.txt") as f:
        for line in f:
            line = line.strip()

            if line.startswith('--extra'):
                current_section = line.split('--extra')[1].strip().lower()
                if current_section not in requirements:
                    requirements[current_section] = []
            elif line and not line.startswith("#"):
                requirements[current_section].append(line)

    return requirements


def get_main_and_extras():
    all_requirements = read_requirements()
    main_requirements = all_requirements.get("main", [])
    extras_require = {
        component: deps
        for component, deps in all_requirements.items()
        if component != "main"
    }
    return main_requirements, extras_require
