//! `senclaw-runtime.json` must parse with the SDK and stay in lockstep with
//! `Cargo.toml`'s version — a manifest shipped with the wrong version number
//! is a package the daemon installs under the wrong directory name.

#[test]
fn manifest_parses_with_the_sdk_and_matches_the_package_version() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/senclaw-runtime.json");
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("reading {path}: {e}"));
    let parsed = sen_runtime_sdk::manifest::RuntimeManifest::parse(&text)
        .expect("senclaw-runtime.json must parse");
    assert!(
        parsed.warnings.is_empty(),
        "unexpected manifest warnings: {:?}",
        parsed.warnings
    );

    let m = parsed.manifest;
    assert_eq!(m.id, "sen-mlx");
    assert_eq!(
        m.version,
        env!("CARGO_PKG_VERSION"),
        "manifest version must track Cargo.toml"
    );
    assert_eq!(m.mode, sen_runtime_sdk::manifest::RunMode::Model);
    assert_eq!(m.slots, vec![sen_runtime_sdk::manifest::Slot::Mlx]);
    assert!(m
        .capabilities
        .contains(&sen_runtime_sdk::manifest::Capability::Chat));
    assert!(m
        .capabilities
        .contains(&sen_runtime_sdk::manifest::Capability::Vision));
}
