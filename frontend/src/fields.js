// Field metadata shared by the form and the workspace highlight overlay.
// Keys match the backend's naming: core names, or "other_fields.<key>".

export const CORE_FIELDS = [
  { name: 'full_name', label: 'Full name' },
  { name: 'licence_number', label: 'Licence number' },
  { name: 'date_of_birth', label: 'Date of birth', kind: 'date' },
  { name: 'date_of_issue', label: 'Date of issue', kind: 'date' },
  { name: 'date_of_expiry', label: 'Date of expiry', kind: 'date' },
  { name: 'address', label: 'Address', kind: 'multiline' },
  { name: 'vehicle_classes', label: 'Vehicle classes' },
  { name: 'issuing_authority', label: 'Issuing authority' },
]

export function humanize(key) {
  const s = key.replace(/_/g, ' ').trim()
  return s.charAt(0).toUpperCase() + s.slice(1)
}

export function fieldDomId(key) {
  return `field-${key.replace('.', '-')}`
}

// Every field of a LicenceData as { key, label, kind, field }.
export function listFields(data) {
  return [
    ...CORE_FIELDS.map((f) => ({ key: f.name, label: f.label, kind: f.kind, field: data[f.name] })),
    ...Object.keys(data.other_fields).map((k) => ({
      key: `other_fields.${k}`,
      label: humanize(k),
      field: data.other_fields[k],
    })),
  ]
}
