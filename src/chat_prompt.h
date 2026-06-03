// ============================================================
// Chat prompt builder (Qwen template) — shared by all backends
// ============================================================
#pragma once
#include "config.h"
#include "model.h"
#include "json_utils.h"
#include <string>
#include <vector>

static inline std::vector<int32_t> build_chat_prompt_tokens(
    const TokenizerConfig& tokenizer, const std::string& json)
{
    struct Msg { std::string role, content; };
    std::vector<Msg> messages;

    size_t pos = json.find("\"messages\"");
    if (pos == std::string::npos) return {};
    pos = json.find('[', pos);
    if (pos == std::string::npos) return {};

    while (pos < json.size()) {
        pos = json.find('{', pos);
        if (pos == std::string::npos || json[pos] == ']') break;
        size_t end_pos = json.find('}', pos);
        if (end_pos == std::string::npos) break;
        std::string piece = json.substr(pos, end_pos - pos + 1);

        std::string role = json_get_string(piece, "role");
        std::string content = json_get_string(piece, "content");
        if (!role.empty() && !content.empty())
            messages.push_back({role, content});

        pos = end_pos + 1;
    }

    if (messages.empty()) return {};

    std::vector<int32_t> tokens;
    for (auto& m : messages) {
        tokens.push_back(IM_START_TOKEN_ID);
        std::string role_str;
        if (m.role == "system") role_str = "system\n";
        else if (m.role == "user") role_str = "user\n";
        else if (m.role == "assistant") role_str = "assistant\n";
        else role_str = m.role + "\n";

        auto role_toks = tokenize(tokenizer, role_str + m.content);
        tokens.insert(tokens.end(), role_toks.begin(), role_toks.end());
        tokens.push_back(IM_END_TOKEN_ID);
        auto nl = tokenize(tokenizer, "\n");
        tokens.insert(tokens.end(), nl.begin(), nl.end());
    }

    tokens.push_back(IM_START_TOKEN_ID);
    auto asst = tokenize(tokenizer, "assistant\n");
    tokens.insert(tokens.end(), asst.begin(), asst.end());

    return tokens;
}

static inline std::vector<int32_t> build_text_prompt_tokens(
    const TokenizerConfig& tokenizer, const std::string& json)
{
    std::string prompt = json_get_string(json, "prompt");
    if (prompt.empty()) {
        std::string sys = json_get_string(json, "system");
        std::string usr = json_get_string(json, "user");
        if (!sys.empty() && !usr.empty()) prompt = sys + "\n" + usr;
        else if (!usr.empty()) prompt = usr;
    }
    if (prompt.empty()) return {};
    return tokenize(tokenizer, prompt);
}
