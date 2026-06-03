// ============================================================
// JSON helpers — shared by all HTTP servers
// ============================================================
#pragma once
#include <string>
#include <sstream>
#include <vector>
#include <cstring>

static inline std::string json_escape(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"') o += "\\\""; else if (c == '\\') o += "\\\\";
        else if (c == '\n') o += "\\n"; else if (c == '\r') o += "\\r";
        else if (c == '\t') o += "\\t"; else o += c;
    }
    return o;
}

struct JsonValue {
    enum Type { Null, Bool, Int, Float, Str, Arr, Obj };
    Type type = Null;
    bool bval = false; int64_t ival = 0; double fval = 0.0;
    std::string sval; std::vector<JsonValue> arr;
    std::vector<std::pair<std::string, JsonValue>> obj;

    static JsonValue null() { JsonValue v; v.type = Null; return v; }
    static JsonValue boolean(bool v) { JsonValue j; j.type = Bool; j.bval = v; return j; }
    static JsonValue integer(int64_t v) { JsonValue j; j.type = Int; j.ival = v; return j; }
    static JsonValue number(double v) { JsonValue j; j.type = Float; j.fval = v; return j; }
    static JsonValue string(const std::string& v) { JsonValue j; j.type = Str; j.sval = v; return j; }
    static JsonValue array() { JsonValue j; j.type = Arr; return j; }
    static JsonValue object() { JsonValue j; j.type = Obj; return j; }
    void push(const JsonValue& v) { arr.push_back(v); }
    void set(const std::string& key, const JsonValue& val) { obj.emplace_back(key, val); }
};

static inline std::string json_render(const JsonValue& v) {
    std::ostringstream o;
    switch (v.type) {
    case JsonValue::Null: o << "null"; break;
    case JsonValue::Bool: o << (v.bval ? "true" : "false"); break;
    case JsonValue::Int: o << v.ival; break;
    case JsonValue::Float: o << v.fval; break;
    case JsonValue::Str:
        o << '"' << json_escape(v.sval) << '"';
        break;
    case JsonValue::Arr:
        o << "["; for (size_t i = 0; i < v.arr.size(); i++) { if (i) o << ","; o << json_render(v.arr[i]); } o << "]"; break;
    case JsonValue::Obj:
        o << "{"; for (size_t i = 0; i < v.obj.size(); i++) { if (i) o << ","; o << '"' << v.obj[i].first << "\":" << json_render(v.obj[i].second); } o << "}"; break;
    }
    return o.str();
}

static inline std::string json_get_string(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return "";
    pos = json.find('"', pos + search.size() + 2);
    if (pos == std::string::npos) return "";
    std::string result; pos++;
    while (pos < json.size() && json[pos] != '"') {
        if (json[pos] == '\\' && pos + 1 < json.size()) { pos++; switch (json[pos]) { case '"': result += '"'; break; case '\\': result += '\\'; break; case 'n': result += '\n'; break; case 't': result += '\t'; break; case 'r': result += '\r'; break; default: result += json[pos]; break; } }
        else result += json[pos];
        pos++;
    }
    return result;
}

static inline float json_get_number(const std::string& json, const std::string& key, float def) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return def;
    pos = json.find(':', pos + search.size());
    if (pos == std::string::npos) return def;
    pos++; while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    if (pos >= json.size()) return def;
    std::string num;
    while (pos < json.size() && (isdigit(json[pos]) || json[pos] == '.' || json[pos] == '-')) num += json[pos++];
    return num.empty() ? def : std::stof(num);
}

static inline int32_t json_get_int(const std::string& json, const std::string& key, int32_t def) {
    return (int32_t)json_get_number(json, key, (float)def);
}
