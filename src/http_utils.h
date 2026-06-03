// ============================================================
// HTTP helpers — shared by all HTTP servers
// ============================================================
#pragma once
#include <string>
#include <sstream>
#include <map>
#include <cstring>

struct HttpRequest { std::string method, path, body; std::map<std::string, std::string> headers; };

static inline HttpRequest parse_http(const std::string& raw) {
    HttpRequest req;
    std::istringstream iss(raw);
    std::string first; std::getline(iss, first);
    auto sp = first.find(' '); if (sp != std::string::npos) req.method = first.substr(0, sp);
    auto sp2 = first.find(' ', sp + 1); if (sp2 != std::string::npos) req.path = first.substr(sp + 1, sp2 - sp - 1);
    std::string line;
    while (std::getline(iss, line)) {
        if (line == "\r" || line.empty()) break;
        auto hs = line.find(':'); if (hs != std::string::npos) {
            std::string k = line.substr(0, hs), v = line.substr(hs + 1);
            if (!v.empty() && v[0] == ' ') v = v.substr(1);
            if (!v.empty() && v.back() == '\r') v.pop_back();
            req.headers[k] = v;
        }
    }
    auto it = req.headers.find("Content-Length"); if (it == req.headers.end()) it = req.headers.find("content-length");
    if (it != req.headers.end()) { int32_t len = std::stoi(it->second); req.body.resize(len); iss.read(&req.body[0], len); }
    return req;
}

static inline std::string http_resp(int32_t code, const std::string& body, const std::string& ct) {
    std::ostringstream o;
    o << "HTTP/1.1 " << code << " " << (code == 200 ? "OK" : code == 400 ? "Bad Request" : "Not Found") << "\r\n";
    o << "Content-Type: " << ct << "\r\nContent-Length: " << body.size() << "\r\nConnection: keep-alive\r\nAccess-Control-Allow-Origin: *\r\n\r\n" << body;
    return o.str();
}

static inline std::string sse(const std::string& json) { return "data: " + json + "\n\n"; }
