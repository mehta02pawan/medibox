#!/usr/bin/env python3

from invoke import Context
from invoke.runners import Result

from typing import List


def sudo_apt_install(
    package_list: List[str],
    password: str,
    use_torsocks=False
        ) -> Result:
    """
    Install apt packages.
    :param package_list: List[str]: List of packages to install.
    :param password: str: Root password (get it with getpass.getpass)
    :param use_torsocks: bool: Use torsocks for installation.
        (Default value = False)

    """
    return Context().sudo(
        command=f"apt install -y {' '.join(package_list)}",
        password=password
        )


def satisfize_dependencies(root_password: str) -> Result:
    """
    Satisfize APT package dependencies.
    :param root_password: str: root password, get it with getpass.getpass

    """
    return sudo_apt_install(
        package_list=["torsocks"],
        password=root_password
        )
