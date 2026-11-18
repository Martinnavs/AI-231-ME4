# Field Sensor Operations Manual

This manual governs deployment, calibration, and fault recovery for the outdoor weather and soil-moisture telemetry nodes installed across the orchard and pump stations. Field technicians should carry the laminated checklist, a calibrated multimeter, and two spare desiccant packs whenever servicing an enclosure.

## Hardware Enclosure and Power

Each telemetry node uses a weatherproof polycarbonate housing mounted on a galvanized steel mast two meters above ground level. Inside the upper tray sits the main controller board, serial identifier NODE-M4-8821, paired with a 12-volt lithium iron phosphate battery pack and a 20-watt monocrystalline solar panel tilted fifteen degrees south.

Before opening the latched cover in damp weather, wipe condensation from the rubber gasket seam so droplets do not fall onto the terminal block. Inspect the inline blade fuse holder on the positive solar lead; a blown 5-amp fuse usually indicates reverse polarity during battery replacement or water ingress in the junction gland. Tightening the cable glands by hand plus a quarter turn with a plastic wrench maintains the IP66 seal without cracking the nylon threads.

When replacing the battery pack, disconnect the solar panel plug first, detach the negative terminal second, and wait ten seconds before pulling the positive lead. Reconnect in reverse order, beginning with the positive lug and finishing with the solar MC4 connector. After reconnecting power, verify that the green charge regulator LED pulses once every two seconds while sunlight hits the panel.

## Soil Probe Calibration

Capacitive soil moisture probes drift slightly after heavy fertilizer application or prolonged drought cracking around the probe sleeve. Quarterly calibration keeps volumetric water content readings accurate within two percentage points across sandy loam and clay plots.

To run a two-point field calibration, first lift the probe cleanly from the sleeve and wipe the fiberglass blade with a dry cotton cloth until no soil film remains. Hold the probe suspended in open air away from metal tools and record the dry raw frequency reading on the handheld terminal. Next, submerge the sensing portion of the blade in a clean bucket of rainwater up to the depth collar, making sure no air bubbles cling to the sensor traces, and record the saturated reading.

Enter both raw frequency values into the maintenance menu under Calibration Slot 2 and press Save to write the updated curve to non-volatile EEPROM. Wait five minutes before reinserting the probe into an irrigated trench, and pack damp screened topsoil gently around the lower collar so air gaps do not bias midday readings downward.

## Radio Mesh Diagnostics

Telemetry packets travel over a 915 MHz LoRa mesh link from orchard nodes to the base gateway mounted on the tool shed roof. Healthy nodes report a received signal strength better than minus 95 dBm and a signal-to-noise ratio above plus 4 dB during calm afternoon checks.

If a node drops three consecutive fifteen-minute reporting windows, inspect the external omnidirectional whip antenna for loose SMA connectors or woodpecker damage on the coaxial jacket. Never power on the radio transceiver without an antenna attached, because reflected power can damage the output amplifier stage within seconds.

When dense mango canopy growth attenuates the direct path to the shed roof, enable single-hop repeater mode on the ridge pole node using configuration flag MESH_RELAY_ON. Record any antenna height adjustments in the mast logbook along with the compass bearing toward the base gateway so future crews do not realign the directional yagi away from the repeater.

## Fault Codes and Recovery

The diagnostic LCD inside the enclosure displays four-character fault codes whenever self-tests fail during startup or hourly polling. Technicians must log the code and timestamp before clearing the error latch.

Fault code ERR-408 indicates that the analog-to-digital converter timed out waiting for the soil temperature thermistor response. Check the three-pin waterproof pigtail connector for green copper corrosion, clean the pins with contact spray, and reboot the board. If ERR-408 persists with a known good spare sensor attached, swap the daughterboard and tag the failed unit for bench inspection.

Fault code ERR-512 means the microSD ring buffer mounted read-only after an unexpected brownout during midnight log compaction. Remove the card only after pressing the unmount button and waiting for the amber activity LED to stay dark for five full seconds. Copy any intact CSV files on a laptop, format the card as FAT32 with 32-kilobyte clusters, and reinsert it before toggling the power switch.

## Seasonal Preventive Maintenance

At the start of the dry season in November, wash dust and bird droppings off every solar panel with mild soap and a soft sponge, then rinse with plain water. Trim tall Napier grass within one meter of the mast base so brush cutters do not nick the buried sensor conduit.

Before heavy monsoon rains arrive in June, swap spent silica gel packets inside every housing for fresh orange indicator packs and apply a thin film of silicone grease to the door O-ring. Tighten all stainless U-bolts on the mast bracket and verify that the lightning arrestor ground rod clamp remains tight and free of rust scale.
