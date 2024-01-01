import unittest
from pprint import pprint

from lichecker import LicenseChecker

# these packages dont define license in setup.py
# manually verified and injected
license_overrides = {
    "filelock": "Unlicense",
    "idna": "BSD",
    "importlib-metadata": "Apache Software License",
    "markdown-it-py": "MIT",
    "mdurl": "MIT",
    "packaging": "Apache Software License, BSD License",
    "kthread": "MIT",
    "yt-dlp": "Unlicense",
    "piper-phonemize": "MIT",
    "pyxdg": "GPL-2.0",
    "ptyprocess": "ISC license",
    "psutil": "BSD3",
    "typing-extensions": "Python Software Foundation License",
    "urllib3": "MIT",
    "zipp": "MIT"
}
# explicitly allow these packages that would fail otherwise
whitelist = ["ovos-skill-installer", "python-dateutil", "numpy"]

# validation flags
allow_nonfree = False
allow_viral = False
allow_unknown = False
allow_unlicense = True
allow_ambiguous = False

pkg_name = "ovos_tts_plugin_piper"


class TestLicensing(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        licheck = LicenseChecker(pkg_name,
                                 license_overrides=license_overrides,
                                 whitelisted_packages=whitelist,
                                 allow_ambiguous=allow_ambiguous,
                                 allow_unlicense=allow_unlicense,
                                 allow_unknown=allow_unknown,
                                 allow_viral=allow_viral,
                                 allow_nonfree=allow_nonfree)
        print("Package", pkg_name)
        print("Version", licheck.version)
        print("License", licheck.license)
        print("Transient Requirements (dependencies of dependencies)")
        pprint(licheck.transient_dependencies)
        self.licheck = licheck

    def test_license_compliance(self):
        print("Package Versions")
        pprint(self.licheck.versions)

        print("Dependency Licenses")
        pprint(self.licheck.licenses)

        self.licheck.validate()
