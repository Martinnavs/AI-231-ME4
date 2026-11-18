# Router Reset Guide

Use this reference when the home fiber router drops packets, stalls during firmware updates, or loses Wi-Fi clients after a power outage. Always check the status LEDs on the front panel before pressing any button.

## Soft Reboot

If the connection slows down or pages fail to load, power-cycle the unit before touching the reset pin. Unplug the barrel power cord from the rear socket, wait thirty full seconds so the capacitors discharge, and plug the cable back in. The optical link LED should turn solid green within ninety seconds.

## Factory Reset

To restore factory defaults when the admin password is lost, locate the recessed pinhole labeled RESET next to the yellow WAN port. While the router is powered on, press and hold the button inside the pinhole with a paperclip for twelve seconds until all four front LEDs blink amber together. Release the pin and allow three minutes for the initial provisioning script to finish.

## Post-Reset Wi-Fi Defaults

After a factory reset, the router broadcasts the default network name VillaNet_Setup_5G and accepts the temporary passphrase printed on the underside label. Connect a laptop over Ethernet port LAN-1 and open http://192.168.8.1 in a browser to set a fresh WPA3 passphrase and disable WPS.
