def main() -> None:
    clip_models = [
        ["rn50", "rn50_ep50", "rn50_ep100"],
        ["rn101", "rn101_ep50"],
        ["vit_b16", "vit_b16_ep50", "vit_b16_ep100"],
        ["vit_b32", "vit_b32_ep50"]
    ]

    open_clip_models = {
        tuple(["rn50", "rn50_ep50", "rn50_ep100"]): ["openai", "yfcc15m", "cc12m"],
        tuple(["rn101", "rn101_ep50"]): ["openai", "yfcc15m"],
        tuple(["vit_b16", "vit_b16_ep50", "vit_b16_ep100"]): ["openai", "laion400m_e31",
                                                              "laion400m_e32", "laion2b_s34b_b88k",
                                                              "datacomp_l_s1b_b8k",
                                                              "commonpool_l_clip_s1b_b8k",
                                                              "commonpool_l_laion_s1b_b8k",
                                                              "commonpool_l_image_s1b_b8k",
                                                              "commonpool_l_text_s1b_b8k",
                                                              "commonpool_l_basic_s1b_b8k",
                                                              "commonpool_l_s1b_b8k"],
        tuple(["vit_b32", "vit_b32_ep50"]): ["openai", "laion400m_e31", "laion400m_e32",
                                             "laion2b_e16", "laion2b_s34b_b79k",
                                             "datacomp_m_s128m_b4k", "commonpool_m_clip_s128m_b4k",
                                             "commonpool_m_laion_s128m_b4k",
                                             "commonpool_m_image_s128m_b4k",
                                             "commonpool_m_text_s128m_b4k",
                                             "commonpool_m_basic_s128m_b4k",
                                             "commonpool_m_s128m_b4k", "datacomp_s_s13m_b4k",
                                             "commonpool_s_clip_s13m_b4k",
                                             "commonpool_s_laion_s13m_b4k",
                                             "commonpool_s_image_s13m_b4k",
                                             "commonpool_s_text_s13m_b4k",
                                             "commonpool_s_basic_s13m_b4k",
                                             "commonpool_s_s13m_b4k"],
        tuple(["roberta-vit-b32"]): ["laion2b_s12b_b32k"],
        tuple(["xlm-roberta-base-vit-b32"]): ["laion5b_s13b_b90k"]
    }

    print("All available clip models with different configs (different number of epochs) (no different pretrained tags!)")
    for model in clip_models:
        print(", ".join(model))

    print()

    print("All available open clip models with different configs (different number of epochs) and different pretrained tags!")
    # Print the dictionary with tuple keys
    for key, value in open_clip_models.items():
        print(f"backbone: {key}, pretrained: {value}")


if __name__ == "__main__":
    main()
