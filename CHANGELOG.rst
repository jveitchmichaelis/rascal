Version 0.3.2
-------------

This version adds quite a lot of small fixes, in particular some nasty bugs in RANSAC which would return duplicate atlas lines,
or incorrect match lists. Error checking has been improved throughout with sanity checks to prevent obviously wrong fits from
sneaking through. Overall fitting should be a bit more robust. This release benefits from several real-world user reports.

We've added a contributor requirement to use pre-commit, which performs some simple linting and runs black to format code prior to commit.

There is a new Atlas class which is the new recommended way to add arc lines. The old methods are still present, but are deprecated
and will be removed in 4.0.

On top of bugfixes, @cylanmarco has done a lot of work to improve automated installation, CI and testing.

* `@jveitchmichaelis <https://github.com/jveitchmichaelis>`__:

  * Added default values for some variables in RANSAC https://github.com/jveitchmichaelis/rascal/commit/83ee2d6eafc2dd58ef791857b531209841ebd20c
  * Some modifications to tolerances in testing to allow for "perfect" synthetic fits
  * Add pre-commit/black code https://github.com/jveitchmichaelis/rascal/commit/8d332879f846c5f7100442a3f5ff1f1a366e5b60
  * Fixes to RANSAC error checking, make checks more consistent https://github.com/jveitchmichaelis/rascal/commit/da4c99878bf3fde457260b8c65f360adf277cbc5
  * New Atlas class https://github.com/jveitchmichaelis/rascal/commit/528162b88534b164be6b7059e9f35073c76eeba5
  * New Plotting library split out from calibrator https://github.com/jveitchmichaelis/rascal/commit/a7f97e40cf01c20d27e1979dc9f8b2d51d5f51f9
  * More logging fixes https://github.com/jveitchmichaelis/rascal/commit/41f1ce74ed28c8e0561e40adebd35791d3a4ce72
  * Matching between peaks and arcs is bijective https://github.com/jveitchmichaelis/rascal/commit/7ed8a20f48825b8d5b9de64e70d9e6ef285365cd
  * Various bugfixes in variables returned by RANSAC

* `@cylammarco <https://github.com/cylammarco>`__:

  * A whole host of improvements to installation and testing https://github.com/jveitchmichaelis/rascal/commit/b59d86329c8dafac54f73c7dce7ebef836fea750
  * More improvements to installation https://github.com/jveitchmichaelis/rascal/commit/bd6d7171dbfc1930b556796ef372f99e1afceb91
  * Fixes to testing https://github.com/jveitchmichaelis/rascal/commit/123d8acbcee8ebe99da4b6e66499aadbdcb4cbb9
  * Bugfix https://github.com/jveitchmichaelis/rascal/commit/b76659bfdda20954ccc70bb689614a043b91dac1
  * Bugfix https://github.com/jveitchmichaelis/rascal/commit/fb881fdedffaf357012d1aa86e2b7082525f27d7

Version 0.3.1
-------------

* `@jveitchmichaelis <https://github.com/jveitchmichaelis>`__:

  * Fix logging propagation https://github.com/jveitchmichaelis/rascal/commit/c72773f51d831dee5068a33df2c419127a2d8490
  * Fixed a bug where matched peaks were not assigned correctly in RANSAC https://github.com/jveitchmichaelis/rascal/commit/24c9c8eca663b665fae6f9b404ec83eee1e8109a
  * Added some checks when matching https://github.com/jveitchmichaelis/rascal/commit/58a920839d2383f206e3819a41d2f528eb293fad
  * dev-stable branch is now tested in CI

Version 0.3.0
-------------

:Date: 28 July 2021

* `@jveitchmichaelis <https://github.com/jveitchmichaelis>`__:

  * Unit testing of SyntheticSpectrum (`#43 <https://github.com/jveitchmichaelis/rascal/issues/43>`__)
  * Automate PyPI publishing (`#40 <https://github.com/jveitchmichaelis/rascal/issues/40>`__)
  * fit() sometimes get stuck (`#38 <https://github.com/jveitchmichaelis/rascal/issues/38>`__, `#35 <https://github.com/jveitchmichaelis/rascal/issues/35>`__)
  * Automate PyPI publishing (`#37 <https://github.com/jveitchmichaelis/rascal/issues/37>`__)
  * Manual removal of pixel-wavelength pairs after fitting (`#30 <https://github.com/jveitchmichaelis/rascal/issues/30>`__)
  * Manual removal of atlas lines (`#29 <https://github.com/jveitchmichaelis/rascal/issues/29>`__)
  * Renamed the primary branch to main (`#28 <https://github.com/jveitchmichaelis/rascal/issues/28>`__)

* `@cylammarco <https://github.com/cylammarco>`__:

  * Match fit() and match_peaks() output format (`#44 <https://github.com/jveitchmichaelis/rascal/issues/44>`__)
  * refine_peaks() handles nan (`#42 <https://github.com/jveitchmichaelis/rascal/issues/42>`__)
  * Include version log (`#41 <https://github.com/jveitchmichaelis/rascal/issues/41>`__)
  * refine_peaks() filters duplicated peaks (`#36 <https://github.com/jveitchmichaelis/rascal/issues/36>`__)
  * Fixed plot_fit() issue with matplotlib (`#32 <https://github.com/jveitchmichaelis/rascal/issues/32>`__)
  * Allow merging of HoughTranform objects (`#31 <https://github.com/jveitchmichaelis/rascal/issues/31>`__)
