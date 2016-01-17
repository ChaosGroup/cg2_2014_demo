#include <string.h>
#include <assert.h>
#include <stdio.h>

#include <sys/types.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <errno.h>

#include <linux/if_ether.h>
#include <linux/if_packet.h>
#include <linux/if.h>
#include <linux/if_arp.h>
#include <arpa/inet.h>

#include "util_eth.hpp"

namespace eth {

// simultaneously init ethernet frame header and socket address structure
// based on http://aschauf.landshut.org/fh/linux/udp_vs_raw/ch01s03.html
bool init_ethhdr_and_saddr(
	const int fd,                 // file descriptor
	const char* const iface_name, // iface name, cstr
	const size_t iface_namelen,   // iface name length, shorter than IFNAMSIZ
	const uint64_t target,        // target mac address, last two MSBs unused
	ethhdr& ethhead,              // output: ethernet frame header
	sockaddr_ll& saddr) {         // output: socket address structure

	assert(IFNAMSIZ	> iface_namelen);

	// pretend we're an established protocol type, lest we get dropped by some self-important filter
	const uint16_t proto = ETH_P_IP;

	// simultaneously fill-in socket address and eth frame header
	memset(&saddr, 0, sizeof(saddr));

	saddr.sll_family   = AF_PACKET;
	saddr.sll_protocol = htons(proto);
	saddr.sll_hatype   = ARPHRD_ETHER;
	saddr.sll_pkttype  = PACKET_OTHERHOST;
	saddr.sll_halen    = ETH_ALEN;

	for (size_t i = 0; i < ETH_ALEN; ++i) {
		saddr.sll_addr[i] = target >> i * 8 & 0xff;
		ethhead.h_dest[i] = target >> i * 8 & 0xff;
	}

	ifreq ifr;
	memcpy(ifr.ifr_name, iface_name, iface_namelen);
	ifr.ifr_name[iface_namelen] = '\0';

	// ioctl about iface index
	if (-1 == ioctl(fd, SIOCGIFINDEX, &ifr)) {
		fprintf(stderr, "error: cannot obtain iface name (errno %s)\n", strerror(errno));
		return false;
	}

	saddr.sll_ifindex  = ifr.ifr_ifindex;

	// ioctl about iface hwaddr
	if (-1 == ioctl(fd, SIOCGIFHWADDR, &ifr)) {
		fprintf(stderr, "error: cannot obtain iface hw addr (errno %s)\n", strerror(errno));
		return false;
	}

	for (size_t i = 0; i < ETH_ALEN; ++i)
		ethhead.h_source[i] = ifr.ifr_hwaddr.sa_data[i];

	ethhead.h_proto = htons(proto);

	return true;
}

} // namespace eth
