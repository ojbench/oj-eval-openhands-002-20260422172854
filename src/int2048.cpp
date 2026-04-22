#include "include/int2048.h"

namespace sjtu {

using std::abs;
using std::complex;
using std::cout;
using std::istream;
using std::max;
using std::min;
using std::ostream;
using std::string;
using std::vector;

static const double PI = acos(-1.0);

static void fft(vector<complex<double>> &a, bool invert) {
  int n = (int)a.size();
  static vector<int> rev;
  static vector<complex<double>> roots{0, 1};
  if ((int)rev.size() != n) {
    int k = __builtin_ctz(n);
    rev.assign(n, 0);
    for (int i = 0; i < n; i++) rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (k - 1));
  }
  if ((int)roots.size() < n) {
    int k = __builtin_ctz(roots.size());
    roots.resize(n);
    while ((1 << k) < n) {
      double ang = 2 * PI / (1 << (k + 1));
      for (int i = 1 << (k - 1); i < (1 << k); ++i) {
        roots[2 * i] = roots[i];
        roots[2 * i + 1] = complex<double>(cos(ang * (2 * i + 1 - (1 << k))), sin(ang * (2 * i + 1 - (1 << k))));
      }
      ++k;
    }
  }
  for (int i = 0; i < n; i++) if (i < rev[i]) std::swap(a[i], a[rev[i]]);
  for (int len = 1; len < n; len <<= 1) {
    for (int i = 0; i < n; i += 2 * len) {
      for (int j = 0; j < len; ++j) {
        complex<double> u = a[i + j];
        complex<double> v = a[i + j + len] * roots[len + j];
        a[i + j] = u + v;
        a[i + j + len] = u - v;
      }
    }
  }
  if (invert) {
    std::reverse(a.begin() + 1, a.end());
    for (int i = 0; i < n; ++i) a[i] /= n;
  }
}

int2048::int2048() : a(1, 0), neg(false) {}

int2048::int2048(long long v) { *this = int2048(); if (v < 0) neg = true, v = -v; while (v) { a.push_back(v % BASE); v /= BASE; } trim(); }

int2048::int2048(const string &s) { read(s); }

int2048::int2048(const int2048 &other) = default;

void int2048::trim() {
  while (!a.empty() && a.back() == 0) a.pop_back();
  if (a.empty()) { a.push_back(0); neg = false; }
}

int int2048::cmp_abs(const int2048 &b) const {
  if (a.size() != b.a.size()) return a.size() < b.a.size() ? -1 : 1;
  for (int i = (int)a.size() - 1; i >= 0; --i) if (a[i] != b.a[i]) return a[i] < b.a[i] ? -1 : 1;
  return 0;
}

int2048 int2048::add_abs(const int2048 &x, const int2048 &y) {
  int2048 r; r.neg = false; r.a.assign(max(x.a.size(), y.a.size()) + 1, 0);
  int carry = 0; size_t i = 0;
  for (; i < x.a.size() || i < y.a.size() || carry; ++i) {
    long long cur = carry;
    if (i < x.a.size()) cur += x.a[i];
    if (i < y.a.size()) cur += y.a[i];
    r.a[i] = (int)(cur % BASE);
    carry = (int)(cur / BASE);
  }
  r.a.resize(i);
  r.trim();
  return r;
}

int2048 int2048::sub_abs(const int2048 &x, const int2048 &y) {
  int2048 r; r.neg = false; r.a.assign(x.a.size(), 0);
  int carry = 0;
  for (size_t i = 0; i < x.a.size(); ++i) {
    long long cur = x.a[i] - carry - (i < y.a.size() ? y.a[i] : 0);
    if (cur < 0) cur += BASE, carry = 1; else carry = 0;
    r.a[i] = (int)cur;
  }
  r.trim();
  return r;
}

static vector<int> naive_conv(const vector<int> &A, const vector<int> &B) {
  vector<long long> tmp(A.size() + B.size());
  for (size_t i = 0; i < A.size(); ++i)
    for (size_t j = 0; j < B.size(); ++j)
      tmp[i + j] += 1LL * A[i] * B[j];
  vector<int> res(tmp.size());
  long long carry = 0;
  for (size_t i = 0; i < tmp.size(); ++i) {
    long long cur = tmp[i] + carry;
    res[i] = (int)(cur % int2048::BASE);
    carry = cur / int2048::BASE;
  }
  while (!res.empty() && res.back() == 0) res.pop_back();
  return res;
}

int2048 int2048::mul_abs_fft(const int2048 &x, const int2048 &y) {
  if ((int)x.a.size() * (int)y.a.size() <= 2000) {
    int2048 r; r.a = naive_conv(x.a, y.a); r.neg = false; r.trim(); return r;
  }
  vector<complex<double>> fa(x.a.begin(), x.a.end()), fb(y.a.begin(), y.a.end());
  int n = 1; while (n < (int)x.a.size() + (int)y.a.size()) n <<= 1;
  fa.resize(n); fb.resize(n);
  fft(fa, false); fft(fb, false);
  for (int i = 0; i < n; ++i) fa[i] *= fb[i];
  fft(fa, true);
  int2048 r; r.a.assign(n, 0); r.neg = false;
  long long carry = 0;
  for (int i = 0; i < n; ++i) {
    long long val = (long long) llround(fa[i].real()) + carry;
    r.a[i] = (int)(val % BASE);
    carry = val / BASE;
  }
  while (carry) { r.a.push_back((int)(carry % BASE)); carry /= BASE; }
  r.trim();
  return r;
}

void int2048::divmod_abs(const int2048 &u, const int2048 &v, int2048 &q, int2048 &r) {
  // simple long division in BASE
  q = int2048(0); r = int2048(0);
  q.a.assign(u.a.size(), 0); q.neg = false; r.neg = false;
  for (int i = (int)u.a.size() - 1; i >= 0; --i) {
    // r = r * BASE + u.a[i]
    if (!(r.a.size() == 1 && r.a[0] == 0)) r.a.insert(r.a.begin(), 0); else r.a[0] = 0;
    r.a[0] = (i < (int)u.a.size() ? r.a[0] + u.a[i] : r.a[0]);
    r.trim();
    int l = 0, rr = BASE - 1, best = 0;
    while (l <= rr) {
      int m = (l + rr) >> 1;
      int2048 t;
      t.a = {m};
      int2048 prod = mul_abs_fft(v, t);
      int cmp;
      // compare prod and r
      cmp = prod.cmp_abs(r);
      if (cmp <= 0) { best = m; l = m + 1; } else rr = m - 1;
    }
    q.a[i] = best;
    if (best) {
      int2048 t; t.a = {best};
      int2048 prod = mul_abs_fft(v, t);
      r = sub_abs(r, prod);
    }
  }
  q.trim(); r.trim();
}

void int2048::read(const string &s) {
  a.clear(); neg = false;
  int i = 0; while (i < (int)s.size() && isspace((unsigned char)s[i])) ++i;
  bool sign = false; if (i < (int)s.size() && (s[i] == '-' || s[i] == '+')) { sign = s[i] == '-'; ++i; }
  int j = (int)s.size() - 1; while (j >= i && isspace((unsigned char)s[j])) --j;
  for (; j >= i; j -= WIDTH) {
    int x = 0; int l = std::max(i, j - WIDTH + 1);
    for (int k = l; k <= j; ++k) x = x * 10 + (s[k] - '0');
    a.push_back(x);
  }
  neg = sign; trim();
}

void int2048::print() {
  if (neg && !(a.size() == 1 && a[0] == 0)) std::cout << '-';
  std::cout << (a.empty() ? 0 : a.back());
  for (int i = (int)a.size() - 2; i >= 0; --i) {
    std::cout << std::setw(WIDTH) << std::setfill('0') << a[i];
  }
}

int2048 &int2048::add(const int2048 &b) {
  if (neg == b.neg) { *this = add_abs(*this, b); this->neg = neg; }
  else {
    int c = cmp_abs(b);
    if (c >= 0) { *this = sub_abs(*this, b); /* sign stays */ }
    else { int2048 t = sub_abs(b, *this); t.neg = b.neg; *this = t; }
  }
  trim(); return *this;
}

int2048 add(int2048 a, const int2048 &b) { return a.add(b); }

int2048 &int2048::minus(const int2048 &b) {
  if (neg != b.neg) { *this = add_abs(*this, b); this->neg = neg; }
  else {
    int c = cmp_abs(b);
    if (c >= 0) { int2048 t = sub_abs(*this, b); t.neg = neg; *this = t; }
    else { int2048 t = sub_abs(b, *this); t.neg = !b.neg; *this = t; }
  }
  trim(); return *this;
}

int2048 minus(int2048 a, const int2048 &b) { return a.minus(b); }

int2048 int2048::operator+() const { return *this; }
int2048 int2048::operator-() const { int2048 t(*this); if (!(t.a.size()==1&&t.a[0]==0)) t.neg = !t.neg; return t; }

int2048 &int2048::operator=(const int2048 &other) = default;

int2048 &int2048::operator+=(const int2048 &b) { *this = add(*this, b); return *this; }
int2048 operator+(int2048 a, const int2048 &b) { a += b; return a; }

int2048 &int2048::operator-=(const int2048 &b) { *this = minus(*this, b); return *this; }
int2048 operator-(int2048 a, const int2048 &b) { a -= b; return a; }

int2048 &int2048::operator*=(const int2048 &b) {
  int2048 r = mul_abs_fft(*this, b); r.neg = (neg != b.neg) && !(r.a.size()==1&&r.a[0]==0); *this = r; return *this;
}
int2048 operator*(int2048 a, const int2048 &b) { a *= b; return a; }

int2048 &int2048::operator/=(const int2048 &b) {
  int2048 ua = *this; ua.neg = false; int2048 vb = b; vb.neg = false; int2048 q, r; divmod_abs(ua, vb, q, r);
  q.neg = (neg != b.neg) && !(q.a.size()==1&&q.a[0]==0);
  // floor division adjustment
  if ((neg != b.neg) && !(r.a.size()==1&&r.a[0]==0)) {
    // if there is remainder, floor needs q-1
    if (!(q.a.size()==1&&q.a[0]==0)) q = minus(q, int2048(1)); else q = int2048(-1);
  }
  *this = q; return *this;
}
int2048 operator/(int2048 a, const int2048 &b) { a /= b; return a; }

int2048 &int2048::operator%=(const int2048 &b) {
  // x % y = x - floor(x/y) * y
  int2048 q = *this / b;
  *this = *this - q * b;
  return *this;
}
int2048 operator%(int2048 a, const int2048 &b) { a %= b; return a; }

istream &operator>>(istream &is, int2048 &x) {
  string s; is >> s; x.read(s); return is;
}

ostream &operator<<(ostream &os, const int2048 &x) {
  if (x.neg && !(x.a.size()==1&&x.a[0]==0)) os << '-';
  os << (x.a.empty() ? 0 : x.a.back());
  for (int i = (int)x.a.size() - 2; i >= 0; --i) {
    os << std::setw(int2048::WIDTH) << std::setfill('0') << x.a[i];
  }
  return os;
}

bool operator==(const int2048 &x, const int2048 &y) { return x.neg==y.neg && x.a==y.a; }
bool operator!=(const int2048 &x, const int2048 &y) { return !(x==y); }
bool operator<(const int2048 &x, const int2048 &y) {
  if (x.neg != y.neg) return x.neg;
  int cmp = x.cmp_abs(y);
  return x.neg ? (cmp > 0) : (cmp < 0);
}
bool operator>(const int2048 &x, const int2048 &y) { return y < x; }
bool operator<=(const int2048 &x, const int2048 &y) { return !(y < x); }
bool operator>=(const int2048 &x, const int2048 &y) { return !(x < y); }

} // namespace sjtu
