HWLOC_DECLSPEC extern const struct hwloc_component hwloc_noos_component;
HWLOC_DECLSPEC extern const struct hwloc_component hwloc_xml_component;
HWLOC_DECLSPEC extern const struct hwloc_component hwloc_synthetic_component;
HWLOC_DECLSPEC extern const struct hwloc_component hwloc_xml_nolibxml_component;
HWLOC_DECLSPEC extern const struct hwloc_component hwloc_windows_component;
HWLOC_DECLSPEC extern const struct hwloc_component hwloc_x86_component;
static const struct hwloc_component * hwloc_static_components[] = {
  &hwloc_noos_component,
  &hwloc_xml_component,
  &hwloc_synthetic_component,
  &hwloc_xml_nolibxml_component,
  &hwloc_windows_component,
  &hwloc_x86_component,
  NULL
};
