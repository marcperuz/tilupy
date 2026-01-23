# CHANGELOG

## [v2.0.1](https://github.com/marcperuz/tilupy/releases/tag/v2.0.1) - 2026-01-23 17:35:09+00:00

Minor updates, add required dependencies and downloading tools.

### Documentation

- README:
  - update README with link to readthedocs ([bf46533](https://github.com/marcperuz/tilupy/commit/bf465335a833333df2569dab984aca2a786ca8cb))

- examples:
  - add missing data ([d11c015](https://github.com/marcperuz/tilupy/commit/d11c015ad71bd054a3f6e37881466623fbab6653))
  - add new examples for documentation with sphinx-galler ([ab16cf4](https://github.com/marcperuz/tilupy/commit/ab16cf48d6be694108b0b24612f19210269d7061))

### Refactor

- examples:
  - remove unused import ([a3d4df7](https://github.com/marcperuz/tilupy/commit/a3d4df7034a16b1c3e7b23d79b4bb7388a93e41a))

## [v2.0.0](https://github.com/marcperuz/tilupy/releases/tag/v2.0.0) - 2025-11-06 15:02:35+00:00

Major version update with tools for Lave2D and Saval2D + processing function for benchmarks.

### Feature

- notations:
  - add getter and setter for name and symbol ([aa4d41e](https://github.com/marcperuz/tilupy/commit/aa4d41efddbdb20904d6c0faab00c82281884f47))

- read:
  - add _nx and _ny to main Results class ([5afb96e](https://github.com/marcperuz/tilupy/commit/5afb96e5d77f2b7f0e76eaf2777bb9d69b007604))
  - change placement of get_profile ([16290e6](https://github.com/marcperuz/tilupy/commit/16290e6c28303ec942dc364e1f83bac488ab7ed7))
  - move get_profile function into Results class ([15732e8](https://github.com/marcperuz/tilupy/commit/15732e880a1a0eef8d365fbc10f62b32a0c5f497))
  - add ALLOWED_MODELS ([c1e5a76](https://github.com/marcperuz/tilupy/commit/c1e5a76e7ecde7e6eb7d0837bff54325661b1ee8))
  - add extract_from_time_step in TemporalResults ([d62630a](https://github.com/marcperuz/tilupy/commit/d62630a53d0a7fe7c66b1d4a72510d035b2e554e))
  - add profile extraction ([d87967b](https://github.com/marcperuz/tilupy/commit/d87967b3530ad5b1252e2b5992c1b44650bcd9d1))
  - reformat TemporalResults0D.plot ([1c76f0e](https://github.com/marcperuz/tilupy/commit/1c76f0e88ab43df6ed67f6aa8a2f2429064776bc)) ([#5](https://github.com/marcperuz/tilupy/pull/5))
  - add features to plot() in TemporalResults1D/2D and StaticResults1D/2D ([3aebd1e](https://github.com/marcperuz/tilupy/commit/3aebd1ea5cfd9e1ff51cda877d529283cf7e1440)) ([#5](https://github.com/marcperuz/tilupy/pull/5))
  - rework class Results ([d416808](https://github.com/marcperuz/tilupy/commit/d416808e4e202ef7a9bc7c16073e621021a733e6)) ([#5](https://github.com/marcperuz/tilupy/pull/5))
  - rename attributes and add properties ([bdefa9f](https://github.com/marcperuz/tilupy/commit/bdefa9fa82484ed4d210528707c3b4db76b074e0))
  - delete variables ([4f0aadb](https://github.com/marcperuz/tilupy/commit/4f0aadbbfb0648d699ba49f651ae78aba02a0674))

- saval2D\read:
  - add output for extraction ([a9c1b1b](https://github.com/marcperuz/tilupy/commit/a9c1b1b0316b08c06ec7271a40dde211a6a55c63))

- lave2D\read:
  - add output for extraction ([c327e76](https://github.com/marcperuz/tilupy/commit/c327e76e0ff17fd7432394e91e0a4d06ce84de77))

- benchmark:
  - change methods name ([4ab573b](https://github.com/marcperuz/tilupy/commit/4ab573bba94435f169c05107b4750ad6794f58e2))
  - add compute_dist_centermass ([e210298](https://github.com/marcperuz/tilupy/commit/e2102984c260f3a403a52189d00d2820a8cf16dd))
  - rename show_multiple_profile ([8e8ae06](https://github.com/marcperuz/tilupy/commit/8e8ae06f3132fe81c8b86b67519ef2ea18a74601))
  - add quantitative comparison between models ([da090fa](https://github.com/marcperuz/tilupy/commit/da090fa8dd8b35f2f475a533f0036a27755f4b99))
  - add show_multiple_profiles ([f85be3a](https://github.com/marcperuz/tilupy/commit/f85be3acf515934ec4bc9a9fafadb71c14538de7)) ([#5](https://github.com/marcperuz/tilupy/pull/5))
  - change class structure ([e25a2c3](https://github.com/marcperuz/tilupy/commit/e25a2c3b8d334c50c24de879b425535b1b36f495)) ([#5](https://github.com/marcperuz/tilupy/pull/5))
  - add flow_velocity_threshold_value ([df79a1e](https://github.com/marcperuz/tilupy/commit/df79a1e238a956247b5088c7bc1132e0a7bfc01b))
  - new features ([fec5e69](https://github.com/marcperuz/tilupy/commit/fec5e695b973c68dc526246bdbbe0c012885a5ae))
  - add benchmark functions ([a16d1f2](https://github.com/marcperuz/tilupy/commit/a16d1f2f3d4db0c4ee9ed79ffdee7321a72d8119))

- analytic_sol:
  - changes in Coussot_Shape ([274a580](https://github.com/marcperuz/tilupy/commit/274a5804cfcf87079f355074b18937352ddba36a))
  - rename Depth_result.show_res to Depth_result.plot ([12c4e87](https://github.com/marcperuz/tilupy/commit/12c4e87df373720224c0e94733faac5ac77a36ba)) ([#5](https://github.com/marcperuz/tilupy/pull/5))
  - remove ShapeResults.show_res ([961778c](https://github.com/marcperuz/tilupy/commit/961778c7e89f6f6f9cd737b1ed1a79b940639961)) ([#5](https://github.com/marcperuz/tilupy/pull/5))
  - add x0 to Chanson's method ([a13c43a](https://github.com/marcperuz/tilupy/commit/a13c43a1ed2cccd8e8ff59ff17f4ad8a1981bbeb))
  - add lateral profil in Shape_result ([ece233f](https://github.com/marcperuz/tilupy/commit/ece233fb0c0c90e37d10aff299eaf14d4d39641b))
  - add x_0 for Mangeney's equation ([287a5fa](https://github.com/marcperuz/tilupy/commit/287a5fac8339f67284b9d0211bf77b1b2c078c6e))
  - add color and linestyle for show_res ([6780127](https://github.com/marcperuz/tilupy/commit/67801273f0cdaae9e9ac30cfa6f4d1a0b646bb18))
  - edit show_res of Depth_result ([207070b](https://github.com/marcperuz/tilupy/commit/207070b2bc8743b6dadbfb4a1ad245939f406373))
  - add Chanson solution to Front_result ([2acaf04](https://github.com/marcperuz/tilupy/commit/2acaf044a2c5fd90124d0b094b5b725f3072bbd7))
  - improve Front_result ([dba3e09](https://github.com/marcperuz/tilupy/commit/dba3e09e7ba31ae386c1122dee49bf91b52e8078))
  - add Chanson's solution ([286e655](https://github.com/marcperuz/tilupy/commit/286e6552fa7c87ead0b2ab66cbd87d19ab167dea))
  - add front position ([c1449b5](https://github.com/marcperuz/tilupy/commit/c1449b5078a4d75cd6c1a2dcb539d2eab87829eb))
  - add coussot shape ([22507f5](https://github.com/marcperuz/tilupy/commit/22507f5ac141b1cfef60ffcc913d015624430af1))
  - fix coussot shape solution ([17e49ae](https://github.com/marcperuz/tilupy/commit/17e49aed0b253a597de7f64a542905464261d722))
  - add shape_res ([7d47851](https://github.com/marcperuz/tilupy/commit/7d47851c851c0f7d02f5bb4ae24ef7d149545f0e))
  - add Mangeley solution ([f09d9e5](https://github.com/marcperuz/tilupy/commit/f09d9e535dba3f81f961150bfaf035c57dedd7b4))
  - add analytic solution ([3c7ec24](https://github.com/marcperuz/tilupy/commit/3c7ec247d1ff1fefb676532a65258bc3e3e588dd))

- models\read:
  - add _read_from_file ([1f89033](https://github.com/marcperuz/tilupy/commit/1f89033e120c643448a9aba44ef03162e67b1de1))

- models.read:
  - compute _tim during init for saval2D and lave2D ([e03ef88](https://github.com/marcperuz/tilupy/commit/e03ef8812e307d95fa9eaf31ec20b566726cb739)) ([#5](https://github.com/marcperuz/tilupy/pull/5))

- saval2D/read:
  - change threshold ([18eb82a](https://github.com/marcperuz/tilupy/commit/18eb82a923d5d41b28d7c722705f6c4b315ceb10))
  - add flowrate threshold ([e7f2898](https://github.com/marcperuz/tilupy/commit/e7f2898347629a76a268d5d4b80ddd8583e94afe))

- shaltop/read:
  - add warning folder_base ([c379f1f](https://github.com/marcperuz/tilupy/commit/c379f1fb37c133d0333ee37ddd85f3b870ab38f9))

- general:
  - feature(shaltop/read): add warning folder_base ([3729866](https://github.com/marcperuz/tilupy/commit/37298663f628cdad43df206f88725d0a875926f0))
  - feature(benchmark): add raise error when time not in recorded pictures ([fe4b8c4](https://github.com/marcperuz/tilupy/commit/fe4b8c4355b72150ede50ec762e6b25d1dcd8d51))
  - feature(read): add get_volume ([8ce29fb](https://github.com/marcperuz/tilupy/commit/8ce29fb04b1e63beba3fc376584de4db8799dc24))

- initdata:
  - create plane surface with mass ([a70e092](https://github.com/marcperuz/tilupy/commit/a70e09214401fd3266c3fa9c554a1b333e75c4eb))

- initsimu:
  - create simulation files ([e566147](https://github.com/marcperuz/tilupy/commit/e566147366766d4212edd59ce5fd20c8ba20e263))

- models:
  - add saval2D ([df5cb23](https://github.com/marcperuz/tilupy/commit/df5cb2353b39c4a18bb5643f79b83e9089417746))

- plot:
  - add option to plot discrete colormap (unique_values) ([627545f](https://github.com/marcperuz/tilupy/commit/627545f086792fba1ea3b1eb920bf135db0a1f4f))

### Bug Fixes

- read:
  - fix variable name in TemporalResults0D.plot() ([2194aae](https://github.com/marcperuz/tilupy/commit/2194aaecedc871c4f9a065f2ed0ecaf6169b4049))
  - colorbar in TemporalResults2D.plot(plot_multiples=True) ([bd79c0f](https://github.com/marcperuz/tilupy/commit/bd79c0f7c7e47397f696d5674c7838da34b9fa06))
  - fix get_output ([473e071](https://github.com/marcperuz/tilupy/commit/473e071b48ec5316c247862820258c0c3df605d1))
  - fix shotgather import ([639f033](https://github.com/marcperuz/tilupy/commit/639f0335032acb134d0df7da03a05ac07b8fffdf)) ([#4](https://github.com/marcperuz/tilupy/pull/4))
  - fix pytopomap import ([4a4d1ba](https://github.com/marcperuz/tilupy/commit/4a4d1baabf36b880c9cc10d1c569b54cb6b2b174)) ([#3](https://github.com/marcperuz/tilupy/pull/3))
  - add return when topography is plotted ([4bf0e46](https://github.com/marcperuz/tilupy/commit/4bf0e46ff94f9d3d81fc32f9c0aee9f6bc2cbf16))
  - adapt cmd for easy plotting/saving topography + minor bug corrections and linting ([b866822](https://github.com/marcperuz/tilupy/commit/b8668228d392a18bea83539f825af02c7bb7adff))

- benchmark:
  - minor fixes ([f423c6d](https://github.com/marcperuz/tilupy/commit/f423c6d1f435ee7df85aad253b6cde7fae794147))
  - fix show_comparison_temporal1D ([eaa0e9f](https://github.com/marcperuz/tilupy/commit/eaa0e9fa074e7b17c80861f137e6eb316d450403))
  - change for get_profile ([30647e8](https://github.com/marcperuz/tilupy/commit/30647e8770df06f53cefacb925e9d003665854ac))
  - fix string configuration in show_multiple_profiles ([3a0ba79](https://github.com/marcperuz/tilupy/commit/3a0ba790ef05b0702aa013915724f0177de8b0de)) ([#5](https://github.com/marcperuz/tilupy/pull/5))
  - fix no output from results ([32afd56](https://github.com/marcperuz/tilupy/commit/32afd56d65eced71a54ad9adc7afb0e77345ad41)) ([#5](https://github.com/marcperuz/tilupy/pull/5))
  - minor problem in extract_velocity_field ([42881ca](https://github.com/marcperuz/tilupy/commit/42881cab83da2a844c6b7e699385ab4bafc5d59b))
  - time selection in extract_velocity_field ([a0d333e](https://github.com/marcperuz/tilupy/commit/a0d333e2f974b0d466413f990879a81b943f5d6a))
  - fix wanted time in profil or surface extraction ([caececd](https://github.com/marcperuz/tilupy/commit/caececd52072f23ec376f5e51900c877f8306a64))

- analytic_sol:
  - minor fixes ([cd410ca](https://github.com/marcperuz/tilupy/commit/cd410ca3ba368afc8aa056690e512480772cf612))
  - fix compute_u ([c5fb8e7](https://github.com/marcperuz/tilupy/commit/c5fb8e791a52b9b14800d78eadd0381695df52bb))
  - fix Dressler's method implementation ([1917431](https://github.com/marcperuz/tilupy/commit/191743133cad4da7fed7784ad4f026ddd6c19308))
  - fix Coussot's solution ([571eab8](https://github.com/marcperuz/tilupy/commit/571eab83be7f0aa2a1e99a30d45c95da07b86f0b))
  - fix analytic solutions ([af69831](https://github.com/marcperuz/tilupy/commit/af698311d20bcfccab4dfd53a0cc24f6b685c477))

- notations:
  - fix get_long_name ([2285793](https://github.com/marcperuz/tilupy/commit/228579343f689164cac768617696528d32656bd9))
  - silence DepreciationWarning ([dea0562](https://github.com/marcperuz/tilupy/commit/dea056206f11f514c225905c0cffe39b7d40b2ab))

- examples:
  - coussot example ([b8b1b17](https://github.com/marcperuz/tilupy/commit/b8b1b17d23e2b0ecb8dcec5b04926e08e79ef095))

- saval2D\read:
  - fix error when invalid output ([6f65c9b](https://github.com/marcperuz/tilupy/commit/6f65c9bdc01747096adf2aa001d0852bdf3b1d5d)) ([#5](https://github.com/marcperuz/tilupy/pull/5))

- exemples:
  - change show_res to plot and remove unused show_res ([fc24e84](https://github.com/marcperuz/tilupy/commit/fc24e84e827c67b38d48928335936107f1134eda)) ([#5](https://github.com/marcperuz/tilupy/pull/5))

- make_topo:
  - correct transition zone in channel() ([431c8f5](https://github.com/marcperuz/tilupy/commit/431c8f5e51c2280db0f8a871eec0e3f234cb50ca))

- test_benchmark:
  - fix tests ([b90ec8b](https://github.com/marcperuz/tilupy/commit/b90ec8b4848f482a679cdf9726f354a901585ddc))

- test_analytic_sol:
  - fix test ([e825459](https://github.com/marcperuz/tilupy/commit/e825459627c279de3f5bb8d4490d6a0223ff53a6))

- analityc_sol:
  - fix H_size in coussot_shape function ([2e289d5](https://github.com/marcperuz/tilupy/commit/2e289d537f5902eecf55aa4f4889c2bbb54b9d0e))

- read_saval2D:
  - fix time extraction ([04e98e0](https://github.com/marcperuz/tilupy/commit/04e98e0b8697ccb33a2f811019674c44d3596cf0))

- initsimu_lave2D:
  - fix initial mass orientation ([fc5bf37](https://github.com/marcperuz/tilupy/commit/fc5bf379fec73d1aca75363a75edc5af9a2708af))

- plot_as_h_u_ritter:
  - fix time import ([057e31a](https://github.com/marcperuz/tilupy/commit/057e31aa149605cb9632473320f8a789d85b6904))

- models.initsimu:
  - change default folder_out ([c225142](https://github.com/marcperuz/tilupy/commit/c2251424a14f27fc175cd89269d62df11cb5d53d))

- models.read:
  - minor fix ([ba37169](https://github.com/marcperuz/tilupy/commit/ba3716953fc7a02cc320f46f539a64c31e6400bb))

- analytic_solution:
  - small changes ([029a591](https://github.com/marcperuz/tilupy/commit/029a591937c3e7761a46b1cefd8630017c8012e3))

- ruff:
  - fix ruff errors ([15d4fda](https://github.com/marcperuz/tilupy/commit/15d4fda73e6240b85461a06f8fdf3f6a5c216399))
  - fix ruff errors ([7ebfca6](https://github.com/marcperuz/tilupy/commit/7ebfca6707edd5897ae2a8d28399a84a29452682))

- utils:
  - close figure in get_contour after contour estimation ([c651b1d](https://github.com/marcperuz/tilupy/commit/c651b1db8bcb4fe2168d5637ebcfda5e54a86247))
  - adapt get_contour to new version of matplotlib ([1dbebb3](https://github.com/marcperuz/tilupy/commit/1dbebb32491e93732423f8d9c0af4db771257af5))

- lave2D:
  - add Results.tim property setter ([378f9bd](https://github.com/marcperuz/tilupy/commit/378f9bdcd11b06ac9e374aa099ad7672ed7710d6))

- plot:
  - pass alpha param from plot_data_on_topo to plot_imshow ([47e4d9e](https://github.com/marcperuz/tilupy/commit/47e4d9ecfbf251edb065ff1debfd956e8b4a1f57))

- calibration:
  - unchain loc and iloc in pd.DataFrame value assignement ([45725dd](https://github.com/marcperuz/tilupy/commit/45725dd6c53ce4bf121cd532c0b6be3ca5eb7ba3))
  - change function calls of previous versions ([42e0801](https://github.com/marcperuz/tilupy/commit/42e0801360b6b013b89c45a2b8064a8e862696d9))

- shaltop:
  - add np.flot32 and np.int32 to types for parameters writing ([c454a9a](https://github.com/marcperuz/tilupy/commit/c454a9ab8cabb530a8ff99c905c1aa74de33ae77))

- notation:
  - silence DepreciationWarning ([0b073bb](https://github.com/marcperuz/tilupy/commit/0b073bb13e5c609dfb39f4bd29958338000f49a5))

### Documentation

- docs:
  - small changes ([fe1ae02](https://github.com/marcperuz/tilupy/commit/fe1ae02ceff7acf0be9d2c197c611a2cef15df1a))
  - edit autoapi templates ([4d2fa76](https://github.com/marcperuz/tilupy/commit/4d2fa769eaa8c2ade8850d1b6a3e877703fec06e))

- models:
  - add corrections to documentation ([1ebf772](https://github.com/marcperuz/tilupy/commit/1ebf7727307b117b79a5266216a5b7604b76ca53))

- tilupy:
  - remove header ([f8f79c0](https://github.com/marcperuz/tilupy/commit/f8f79c09b71f4f025c85366619cd3b264c4f7405))

- read:
  - add corrections to documentation ([0766da9](https://github.com/marcperuz/tilupy/commit/0766da93b960ef41816d76c42c546711966f0d71))
  - add documentation ([d8ee3d6](https://github.com/marcperuz/tilupy/commit/d8ee3d690eae8b2cc3d36682212b407540fe039c))

- make_topo:
  - add corrections to documentation ([e2821be](https://github.com/marcperuz/tilupy/commit/e2821bebb69ef3b1f5a82e16b904a3bdcd60d1b8))
  - reformat documentation ([9bec410](https://github.com/marcperuz/tilupy/commit/9bec4103ecc63cb54062e5f48fea0b0671b2ca41)) ([#5](https://github.com/marcperuz/tilupy/pull/5))

- make_mass:
  - add corrections to documentation ([174f5c8](https://github.com/marcperuz/tilupy/commit/174f5c87c2f2d87e1ed52e5a5db9ed69780cb4c2))
  - reformat documentation ([a4bf57f](https://github.com/marcperuz/tilupy/commit/a4bf57f4cc0c65b45d64fdb0c8462d6e220019f2)) ([#5](https://github.com/marcperuz/tilupy/pull/5))

- initsimus:
  - add corrections to documentation ([3cc7407](https://github.com/marcperuz/tilupy/commit/3cc7407d333a178faf5c7f2ed81cfb53516a7c86))

- initdata:
  - add corrections to documentation ([94b6f3e](https://github.com/marcperuz/tilupy/commit/94b6f3e79d1bbcabc3bb29957ab4fd7e128c2da1))
  - add documentation ([9052c9b](https://github.com/marcperuz/tilupy/commit/9052c9bddcc35c6bb6c9f52bbde526c9ea9d261d)) ([#5](https://github.com/marcperuz/tilupy/pull/5))

- benchmark:
  - add corrections to documentation ([fa98845](https://github.com/marcperuz/tilupy/commit/fa98845843efd2424ae4ee1e4b66495b634bc5cd))
  - add details in documentation ([49521f1](https://github.com/marcperuz/tilupy/commit/49521f107f23098db3326dcca6d679d0a4d82074)) ([#5](https://github.com/marcperuz/tilupy/pull/5))

- analytic_sol:
  - add corrections to documention ([2b19967](https://github.com/marcperuz/tilupy/commit/2b19967684f4aae6b0bc6793f57f5eaecd367818))
  - edit documentation ([7335e8b](https://github.com/marcperuz/tilupy/commit/7335e8b4ddca4f0042128de48c278d3118b0dee2)) ([#5](https://github.com/marcperuz/tilupy/pull/5))

- notations:
  - add documentation ([4dfd482](https://github.com/marcperuz/tilupy/commit/4dfd482c46a4a995c57752eb87528642fdc9cac1))

- utils:
  - add documenation ([9512e5a](https://github.com/marcperuz/tilupy/commit/9512e5add98be0a53014b009f3d056ba08e2361e))

- download_data:
  - add documentation ([cf3ec58](https://github.com/marcperuz/tilupy/commit/cf3ec5895bf188737fa7d3945040218e8b0d14d5)) ([#5](https://github.com/marcperuz/tilupy/pull/5))

- plot:
  - add documentation ([48adbfd](https://github.com/marcperuz/tilupy/commit/48adbfdd359b239e7d43f50f66e8ce08a41db4b3)) ([#5](https://github.com/marcperuz/tilupy/pull/5))

- raster:
  - add documentation ([c4a8062](https://github.com/marcperuz/tilupy/commit/c4a80629058277b39cfdd64641d226ff2ca91fd1)) ([#5](https://github.com/marcperuz/tilupy/pull/5))

- models.lave2d:
  - add documentation ([05bbe7b](https://github.com/marcperuz/tilupy/commit/05bbe7b3cc50c686a588ab9c4b34547eeefe94ba)) ([#5](https://github.com/marcperuz/tilupy/pull/5))

- models.saval2d:
  - add documentation ([fa9bfb6](https://github.com/marcperuz/tilupy/commit/fa9bfb6de2548e29034e524e3eed6d4ac0079cea)) ([#5](https://github.com/marcperuz/tilupy/pull/5))

- models.shaltop:
  - add documentation ([5992431](https://github.com/marcperuz/tilupy/commit/59924311790d29209438636c96002abab8cfa948)) ([#5](https://github.com/marcperuz/tilupy/pull/5))

- initsimu & read:
  - complete documentation ([e00a06b](https://github.com/marcperuz/tilupy/commit/e00a06b0d6bef715a70bc4b813480cc881b6b93e)) ([#5](https://github.com/marcperuz/tilupy/pull/5))

- conf:
  - add intersphinx ([c98efb7](https://github.com/marcperuz/tilupy/commit/c98efb7f6968e6d828c4d780e9307170376bf0a6)) ([#5](https://github.com/marcperuz/tilupy/pull/5))

- models/initsimu:
  - add and edit docstring ([a248121](https://github.com/marcperuz/tilupy/commit/a2481210706420560235f005437319b33cd7dd19))

- shaltop/initsimu:
  - add docstring for functions ([2a3d73c](https://github.com/marcperuz/tilupy/commit/2a3d73c45245f250781e098451e77148d9fcd7b1))

- lave2D/initsimu:
  - add docstring for functions ([3578bef](https://github.com/marcperuz/tilupy/commit/3578bef114934456e6843a74a8760530c4ace716))

- readme:
  - edit example ([b6a82d6](https://github.com/marcperuz/tilupy/commit/b6a82d627891d37fb5e4ce86fea64b6610b58f1a))
  - edit example ([148e841](https://github.com/marcperuz/tilupy/commit/148e841a4755d40d3d7d56783ad980284b99f127))

- index:
  - change pytopomap ref to tilupy ([b09a95b](https://github.com/marcperuz/tilupy/commit/b09a95b69301b24f02adc0ad56f0998d5c1d03b3))

- README:
  - add anaconda badges and update table of contents ([8a03d0e](https://github.com/marcperuz/tilupy/commit/8a03d0ee9a229e520b8ba8abffeb5b0638f7e919))

### Refactor

- tilupy:
  - fix ruff changes ([25f1329](https://github.com/marcperuz/tilupy/commit/25f1329f2db96d6051c403bae6ec1e3ca24a7830))

- test:
  - change test_shaltop to test_shaltop_initsimus ([7d51053](https://github.com/marcperuz/tilupy/commit/7d510534a2794564b4a51cd46ed66d91a7c62b63)) ([#5](https://github.com/marcperuz/tilupy/pull/5))
  - change test_lave2d to test_lave2d_initsimus ([e2589ba](https://github.com/marcperuz/tilupy/commit/e2589ba96ef0f2132a8aff89e2eab2a6bc92e62f)) ([#5](https://github.com/marcperuz/tilupy/pull/5))
  - change test_generate_data to test_make_topo ([141d2d3](https://github.com/marcperuz/tilupy/commit/141d2d3115a924eb2c24efc80178ceb931a7eba5)) ([#5](https://github.com/marcperuz/tilupy/pull/5))

- benchmark:
  - ruff change ([70e2b92](https://github.com/marcperuz/tilupy/commit/70e2b920d481161f03b36e5f0607194ddc99e2a4)) ([#5](https://github.com/marcperuz/tilupy/pull/5))
  - ruff change ([ee4f17f](https://github.com/marcperuz/tilupy/commit/ee4f17f907c8e749412894e0c8f26260896635ca)) ([#5](https://github.com/marcperuz/tilupy/pull/5))
  - remove import cm ([114f8cf](https://github.com/marcperuz/tilupy/commit/114f8cf2bbcc8e5f5d237d4fc13e69b0158b4133))
  - replace cm by plt ([29f4087](https://github.com/marcperuz/tilupy/commit/29f40878780735363a84ed8da90cd501261f8dc7))
  - ruff changes ([cda16d2](https://github.com/marcperuz/tilupy/commit/cda16d20d35b6746dc5205cc3b0a7dbe4cfab444))
  - ruff corrections ([d319212](https://github.com/marcperuz/tilupy/commit/d31921208e1a91d4240496a6d8d9af273ab9b4dc))

- initdata:
  - ruff changes ([4a85c57](https://github.com/marcperuz/tilupy/commit/4a85c576bb6b62c4577e73539840dae42813a9c7)) ([#5](https://github.com/marcperuz/tilupy/pull/5))

- test_read_from_simu:
  - change folder_base to folder ([1770705](https://github.com/marcperuz/tilupy/commit/1770705aa933771f1637a1272fee56f13c7a1d18))

- cmd:
  - change variable folder_base to folder ([2ad707c](https://github.com/marcperuz/tilupy/commit/2ad707c39f564588dad2bfa8046b9764843887cc))

- test_benchmark:
  - change name folder_base to folder ([16774ce](https://github.com/marcperuz/tilupy/commit/16774ced1ad1cd5354b85f93f7863fc29ff5ba74))

- models/shaltop:
  - change folder_base to folder ([4cea88c](https://github.com/marcperuz/tilupy/commit/4cea88ca15d1aac428690700ba85624e77445afb))

- make_topo:
  - update scipy method ([f24e500](https://github.com/marcperuz/tilupy/commit/f24e5005762d07312c5c0a92b04c78055701b1c6))

- ruff:
  - correct ruff error raise ([c72433c](https://github.com/marcperuz/tilupy/commit/c72433cffcc9707498b8547f45a3b1252b82a726))

- saval2D:
  - ruff corrections ([c380611](https://github.com/marcperuz/tilupy/commit/c3806113e4ce47a281567f5133645eedf5cd8d8e))

- analytic_sol:
  - rename function ([2d21c65](https://github.com/marcperuz/tilupy/commit/2d21c6584048ea8fe2d4b59f8494a36259266318))
  - change int to float ([94edab1](https://github.com/marcperuz/tilupy/commit/94edab19fac92d0e94eac16514567d8e653d5899))
  - change Mangeney equation ([42022c9](https://github.com/marcperuz/tilupy/commit/42022c90a3efcbab255ecf8dde0ff3968202d950))
  - fix ruff errors ([3a9a6e1](https://github.com/marcperuz/tilupy/commit/3a9a6e142c46034d4fc8ab388adc9086d39f1da5))

- read:
  - change folder output variable name for save ([2572309](https://github.com/marcperuz/tilupy/commit/257230961986059d39893c5e2e8d64c278da2968))

## [v1.3.0](https://github.com/marcperuz/tilupy/releases/tag/v1.3.0) - 2024-08-29 11:51:45

Minor bug fix + functions to determine the position of the center of mass.

### Feature

- read:
  - add z coordinate to center of mass position ([7e5840c](https://github.com/marcperuz/tilupy/commit/7e5840cfcf149a07ec97519265dc4630e3953fd0)) ([#2](https://github.com/marcperuz/tilupy/pull/2))
  - add function to get center of mass position ([f03777e](https://github.com/marcperuz/tilupy/commit/f03777eb23048fca316bfbde7badc00ce1090a64)) ([#2](https://github.com/marcperuz/tilupy/pull/2))

### Bug Fixes

- make_topo:
  - change cumptrapz to cumulative_trapezoid ([fc9a867](https://github.com/marcperuz/tilupy/commit/fc9a867bd7a96d1908b0a4ab2d78b0f1560444de))

- plot:
  - correct plot_topography function ([4d83e70](https://github.com/marcperuz/tilupy/commit/4d83e70866d0dffb51f644832d3925972b0b701c))
  - correct determination of contour intervals ([2ed4667](https://github.com/marcperuz/tilupy/commit/2ed46672771d11bafddfe2b0c8040696fac6c187))
  - fix contour level determination when ndv is given ([80b78bc](https://github.com/marcperuz/tilupy/commit/80b78bc5ea299790f7f5184e0b1b53df0e24a952))

### Refactor

- read:
  - deal with depreciation warnings ([da92c18](https://github.com/marcperuz/tilupy/commit/da92c183871f0d777637932507beb77188df0f3a))

## [v1.2.0](https://github.com/marcperuz/tilupy/releases/tag/v1.2.0) - 2024-06-07 10:08:07

Function to set an initial mass in Lave2D simulation

### Feature

- lave2D:
  - add function for setting initial mass ([953f07a](https://github.com/marcperuz/tilupy/commit/953f07a76553eca79bf084c9c7478d439c6914ca))

### Documentation

- README:
  - Add references for Lave2D in README ([ed1103c](https://github.com/marcperuz/tilupy/commit/ed1103cb9851340cd49adbb0a08edebf2d68e85c))

## [v1.1.0](https://github.com/marcperuz/tilupy/releases/tag/v1.1.0) - 2024-06-05 16:42:11

Add a module for LAVE2D simulations.
Minor bug corrections and improvements.

### Feature

- lave2D:
  - add read module for lave2D simulations ([8a6f35f](https://github.com/marcperuz/tilupy/commit/8a6f35fb22b291e4854dc15033ba6466af6d07e0)) ([#1](https://github.com/marcperuz/tilupy/pull/1))
  - add functions to prepare Lave2D simulations ([cb8fcd0](https://github.com/marcperuz/tilupy/commit/cb8fcd0e8f1b5124867eb57c5a8a35efee75cc2f)) ([#1](https://github.com/marcperuz/tilupy/pull/1))
  - add method for deterimining cell edges from coordinates ([a696a31](https://github.com/marcperuz/tilupy/commit/a696a31ba9cfcd877e9ee50e1ff301d2f516cd54)) ([#1](https://github.com/marcperuz/tilupy/pull/1))
  - add subpackage for processing LAVE2D ([a69e187](https://github.com/marcperuz/tilupy/commit/a69e1872bc3a4f53312ac1dd0fe08a4d7f6fb271)) ([#1](https://github.com/marcperuz/tilupy/pull/1))

- plot:
  - add automatic shading function ([9c27f91](https://github.com/marcperuz/tilupy/commit/9c27f91021d619d455439fc4fa1a2de78aae165f))
  - change behaviour of saving plots ([9823222](https://github.com/marcperuz/tilupy/commit/98232221f7290693d5451ad9a30b6da305ed5fca))

### Bug Fixes

- lave2D:
  - correct axis coordinates for Modelling domain ([da28b49](https://github.com/marcperuz/tilupy/commit/da28b49f9ab02a3782d244703a2c30b9db5de052)) ([#1](https://github.com/marcperuz/tilupy/pull/1))

- read:
  - get time indexes that must be saved for TemporalResults ([77e83bb](https://github.com/marcperuz/tilupy/commit/77e83bbf589fccb35ebbf968d1d2cdbba8bf5944)) ([#1](https://github.com/marcperuz/tilupy/pull/1))
  - check existence of folder_out attribute ([6ac28f2](https://github.com/marcperuz/tilupy/commit/6ac28f2e84f8009850e39692b69d0252af50476c)) ([#1](https://github.com/marcperuz/tilupy/pull/1))
  - fix path merging ([a7ddf7d](https://github.com/marcperuz/tilupy/commit/a7ddf7d11fe16174452e645704d39e07bb6f0787))
  - correct numpy gradient usage ([2b94bef](https://github.com/marcperuz/tilupy/commit/2b94bef0d447945bfa41319f56906367663845e0))

- plot:
  - set colormap when only nan in data ([0b7d818](https://github.com/marcperuz/tilupy/commit/0b7d8180ecc861af0ecab86dcc0dc6e08e05a621)) ([#1](https://github.com/marcperuz/tilupy/pull/1))
  - remove non necessary tight_layout and assertion and  and plot saving options ([e54a82d](https://github.com/marcperuz/tilupy/commit/e54a82df64c92dd92ce2eb08d35358b8514150b2))

- shaltop.read:
  - correct file identifier name ([2f4a0ca](https://github.com/marcperuz/tilupy/commit/2f4a0cafd10c1ea52078b559bca5a9f3d6effd53))
  - ignore blank lines when reading parameters ([328fe03](https://github.com/marcperuz/tilupy/commit/328fe03aa5c95b3607ae6b7e2ee7ef8a967d890f))
  - check for the existence of time_forces.d ([f86d04a](https://github.com/marcperuz/tilupy/commit/f86d04a0343b857344bfab0c3cd7fef896de8017))

- shaltop:
  - correct loading of velocity for shaltop simulations ([175f392](https://github.com/marcperuz/tilupy/commit/175f3925e70410c3d016fb38e3ed3b2ca0a8dd0a))

- gray99:
  - correct setup for gray99 experiment ([bc641b8](https://github.com/marcperuz/tilupy/commit/bc641b88ad7cc516db73176c20f3cbe6bfab1e65))

### Documentation

- README:
  - add installation instruction from conda-forge ([93eccdc](https://github.com/marcperuz/tilupy/commit/93eccdc51b2471abc5104adfc356d9171410fa81))

### Refactor

- plot:
  - change default valueof from_file in Results.read ([b72932e](https://github.com/marcperuz/tilupy/commit/b72932e92820a2ff951963b137874028c8e6135c))

## [v1.0.0](https://github.com/marcperuz/tilupy/releases/tag/v1.0.0) - 2024-02-01 13:40:53

Management of spatial and temporal integration of simulation results. Major improvements include 
 -  Automatic and intelligent management of notations (symbols, litteral names, units)
 - Automatic generation of notations when results are processed (e.g. time or spatial integrals)
 - Generic plot functions for various result types, in particular time dependent results (line plots and "shotgather" type plots)
 - Simplified and intuitive recovery of simulations results from strings, including reading code-specific outputs

## [v0.1.5](https://github.com/marcperuz/tilupy/releases/tag/v0.1.5) - 2024-01-04 12:54:39

New release with minor bugs/style corrections

### Feature

- data:
  - add Gray99 topography and mass as data ([b02e092](https://github.com/marcperuz/tilupy/commit/b02e09276e15ffcdefc486b497b8f64d64021da5))

- make_mass:
  - add module for initial synthetic mass generation ([ca39f1b](https://github.com/marcperuz/tilupy/commit/ca39f1b26d6c69624eb568180f67fe89acf0cb22))

- plot:
  - add automatic calculation of contour line intervals ([934b1ce](https://github.com/marcperuz/tilupy/commit/934b1ce5f47f30956d2572993ec4e587f1bc519b))

- make_topo:
  - add function to generate synthetic channels. ([e8d5de0](https://github.com/marcperuz/tilupy/commit/e8d5de0b05cdcde1c30f54e0dff852329395e64c))
  - generate topography from the experiment of Gray99 ([6005075](https://github.com/marcperuz/tilupy/commit/6005075c17ae72ff53833a60ca2e5ed11918e76d))

### Bug Fixes

- tests:
  - fix indentation error in shaltop test function ([53c9249](https://github.com/marcperuz/tilupy/commit/53c92493e2efa67d379675565ce53e4aee9d1971))

- calibration:
  - fix old version of module import not corrected ([1af2a4a](https://github.com/marcperuz/tilupy/commit/1af2a4aec3c63e0663c45fb349d8b423d2cee2b6))

- test:
  - change test assertion in test_generated_data.py ([b132750](https://github.com/marcperuz/tilupy/commit/b13275075b6908fb5820879fc4daca857f6e2ad8))

### Documentation

- README:
  - correct typo ([9985c8f](https://github.com/marcperuz/tilupy/commit/9985c8f9559b67b6da7b86231601753e9f7755cc))

## [v0.1.4](https://github.com/marcperuz/tilupy/releases/tag/v0.1.4) - 2023-07-20 14:17:06

Compatibility with python 3.8 to 3.11 is ensured and tested.

### Bug Fixes

- test_shaltop:
  - replace os.rmdir by shutil.rmtree ([2c8c7e6](https://github.com/marcperuz/tilupy/commit/2c8c7e6e7b86247d1016b48597da354592c66ba6))

- tilupy.cmd:
  - Use old behaviour of glob.glob, without root_dir parameter ([223dedb](https://github.com/marcperuz/tilupy/commit/223dedb4dd0996e8eee42a61c6ad6f50b6171b47))

### Documentation

- version:
  - increase version to 0.1.4 ([55ed352](https://github.com/marcperuz/tilupy/commit/55ed3522ac8b1d080114630e02faf2f341d89866))

## [v0.1.3](https://github.com/marcperuz/tilupy/releases/tag/v0.1.3) - 2023-07-20 12:52:45

Minimum python required is >=3.10, needed for glob module

## [v0.1.2](https://github.com/marcperuz/tilupy/releases/tag/v0.1.2) - 2023-07-20 12:33:17

Realease first version to PyPi

### Documentation

- version:
  - change version for testing pip install in test-pypi ([8e8398c](https://github.com/marcperuz/tilupy/commit/8e8398c755cfb946470cc36d5e3f94ef777d5802))
  - Change version to test download to test-pypi ([97c9013](https://github.com/marcperuz/tilupy/commit/97c9013581fb6788efdf9272538ae89a5de3bd73))

## [v0.1.0](https://github.com/marcperuz/tilupy/releases/tag/v0.1.0) - 2023-07-20 09:01:35

*No description*

### Documentation

- pyproject.toml:
  - <subject>Add correct license classifier ([9687e59](https://github.com/marcperuz/tilupy/commit/9687e590dea3b9ca0f0fb7ea9b4a4d52bbbefd28))

\* *This CHANGELOG was automatically generated by [auto-generate-changelog](https://github.com/BobAnkh/auto-generate-changelog)*
