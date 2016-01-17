#ifndef util_eth_H__
#define util_eth_H__

#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>

#include <sys/types.h>
#include <sys/socket.h>

#include <linux/if_ether.h>
#include <linux/if_packet.h>
#include <linux/if.h>
#include <arpa/inet.h>

namespace eth {

// eth frame geometry, sans preamble and FCS/CRC
static const size_t frame_header_len = ETH_HLEN;    // 14 octets header
static const size_t frame_min_size = ETH_ZLEN;      // minimal frame size (14 octets header + 46 octets payload)
static const size_t frame_max_size = ETH_FRAME_LEN; // full frame size (14 octets header + 1500 octets payload)
static const size_t packet_size = ETH_DATA_LEN;     // 1500 octets payload in the full frame
static const size_t packet_min_size = ETH_ZLEN - ETH_HLEN;

class non_copyable
{
	non_copyable(const non_copyable&) {}
	non_copyable& operator =(const non_copyable&) { return *this; }

public:
	non_copyable() {}
};

template < typename T, class FTOR_T >
class scoped : public non_copyable, private FTOR_T {
	T m;

public:
	scoped(T arg)
	: m(arg) {
	}

	~scoped() {
		FTOR_T::operator()(m);
	}

	operator T () const {
		return m;
	}
};

struct generic_free {
	void operator()(void* const ptr) const {
		if (0 != ptr)
			free(ptr);
	}
};

struct close_file_descriptor {
	void operator ()(const int fd) const {
		if (0 <= fd)
			close(fd);
	}
};

// simultaneously init ethernet frame header and socket address structure
// based on http://aschauf.landshut.org/fh/linux/udp_vs_raw/ch01s03.html
bool init_ethhdr_and_saddr(
	const int fd,                 // file descriptor
	const char* const iface_name, // iface name, cstr
	const size_t iface_namelen,   // iface name length, shorter than IFNAMSIZ
	const uint64_t target,        // target mac address, last two MSBs unused
	ethhdr& ethhead,              // output: ethernet frame header
	sockaddr_ll& saddr);

} // namespace eth

#endif // util_eth_H__
