use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        // for tensor_name in safetensor.names() {
        //     println!("Tensor name: {}", tensor_name);
        // }

        use safetensors::tensor::Dtype;
        let get_tensor = |name: &str| {
            let view = safetensor.tensor(name).unwrap();
            assert_eq!(view.dtype(), Dtype::F32);

            // Cast bytes to f32 vec
            let data = unsafe {
                let data = view.data().to_vec();
                let ptr = data.as_ptr() as *const f32;
                let len = data.len() / std::mem::size_of::<f32>();
                std::mem::forget(data);
                Vec::from_raw_parts(ptr as *mut f32, len, len)
            };

            Tensor::<f32>::new(data, &view.shape().to_vec())
        };

        let layers = config.num_hidden_layers;
        LLamaParams {
            embedding_table: get_tensor("lm_head.weight"),
            rms_att_w: (0..layers)
                .map(|i| get_tensor(format!("model.layers.{}.input_layernorm.weight", i).as_str()))
                .collect(),
            wq: (0..layers)
                .map(|i| get_tensor(format!("model.layers.{}.self_attn.q_proj.weight", i).as_str()))
                .collect(),
            wk: (0..layers)
                .map(|i| get_tensor(format!("model.layers.{}.self_attn.k_proj.weight", i).as_str()))
                .collect(),
            wv: (0..layers)
                .map(|i| get_tensor(format!("model.layers.{}.self_attn.v_proj.weight", i).as_str()))
                .collect(),
            wo: (0..layers)
                .map(|i| get_tensor(format!("model.layers.{}.self_attn.o_proj.weight", i).as_str()))
                .collect(),
            rms_ffn_w: (0..layers)
                .map(|i| {
                    get_tensor(
                        format!("model.layers.{}.post_attention_layernorm.weight", i).as_str(),
                    )
                })
                .collect(),
            w_up: (0..layers)
                .map(|i| get_tensor(format!("model.layers.{}.mlp.up_proj.weight", i).as_str()))
                .collect(),
            w_gate: (0..layers)
                .map(|i| get_tensor(format!("model.layers.{}.mlp.gate_proj.weight", i).as_str()))
                .collect(),
            w_down: (0..layers)
                .map(|i| get_tensor(format!("model.layers.{}.mlp.down_proj.weight", i).as_str()))
                .collect(),
            rms_out_w: get_tensor("model.norm.weight"),
            lm_head: get_tensor("lm_head.weight"),
        }
    }
}
