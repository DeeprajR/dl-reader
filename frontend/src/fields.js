// Field metadata shared by the form and the workspace highlight overlay.
//
// The form shows exactly nine fields: the eight core fields below plus one "Other relevant
// information" field. That ninth field is an editable view of LicenceData.other_fields, one
// "Label: value" line per item; the backend keeps each item separately, with its own source,
// confidence and highlight.

// The eight fixed fields, in display order. `name` is the key in the API's data and `label` is
// the text the user sees. `kind` picks the input: a date box, or a multi-line box.
export const CORE_FIELDS = [
  { name: 'full_name', label: 'Full Name' },
  { name: 'licence_number', label: 'Driving Licence Number' },
  { name: 'date_of_birth', label: 'Date of Birth', kind: 'date' },
  { name: 'date_of_issue', label: 'Date of Issue', kind: 'date' },
  { name: 'date_of_expiry', label: 'Date of Expiry', kind: 'date' },
  { name: 'address', label: 'Address', kind: 'multiline' },
  { name: 'vehicle_classes', label: 'Vehicle/Class of Licence' },
  { name: 'issuing_authority', label: 'Issuing Authority' },
]

// The ninth field, which holds everything else found on the licence.
export const OTHER_FIELD = { name: 'other_fields', label: 'Other relevant information' }

// "blood_group" -> "Blood group".
export function humanize(key) {
  const s = key.replace(/_/g, ' ').trim()
  return s.charAt(0).toUpperCase() + s.slice(1)
}

// The HTML id of a field's input. Clicking a highlight on the document uses it to find, scroll
// to and focus the field. (Dots are not safe in ids: "other_fields.blood_group".)
export function fieldDomId(key) {
  return `field-${key.replace('.', '-')}`
}

// Per-class validity items, e.g. "lmv_date_of_issue" -> "LMV · Date of issue".
const CLASS_DATE_KEY = /^(.+?)_(date_of_issue|valid_till)$/

export function otherLabel(key) {
  const m = key.match(CLASS_DATE_KEY)
  if (!m) return humanize(key)
  return `${m[1].replace(/_/g, ' ').toUpperCase()} · ${m[2] === 'date_of_issue' ? 'Date of issue' : 'Valid till'}`
}

// Label -> backend key; reverses otherLabel/humanize ("LMV · Date of issue" -> "lmv_date_of_issue").
function toKey(label) {
  return label
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .slice(0, 64)
}

// Items of other_fields that have a value, as { key, label, field }.
export function otherItems(otherFields) {
  return Object.entries(otherFields)
    // Items without a value are not shown.
    .filter(([, field]) => field.value != null && field.value !== '')
    .map(([key, field]) => ({ key, label: otherLabel(key), field }))
}

// The text shown in the "Other relevant information" field.
export function composeOther(otherFields) {
  return otherItems(otherFields)
    // Each item goes on one line, so a line break inside a value becomes a comma.
    .map(({ label, field }) => `${label}: ${field.value.replace(/\s*\n\s*/g, ', ')}`)
    .join('\n')
}

// Map the edited text back to other_fields. Existing items keep their source, confidence and
// highlight; new lines become new items; deleted lines are removed. A line without "Label: "
// is kept under "Additional information".
export function parseOther(text, previous) {
  // Look up each existing item by the label it is shown under, e.g. "blood group" -> "blood_group".
  const byLabel = new Map(Object.keys(previous).map((key) => [otherLabel(key).toLowerCase(), key]))
  const result = {}
  for (const raw of text.split('\n')) {
    const line = raw.trim()
    if (!line) continue
    // Split the line at the first ": " into a label and a value.
    const split = line.indexOf(': ')
    const label = split > 0 ? line.slice(0, split).trim() : 'Additional information'
    const value = split > 0 ? line.slice(split + 2).trim() : line
    // A known label keeps its existing key. A new label gets a key made from its text.
    const key = byLabel.get(label.toLowerCase()) ?? (toKey(label) || 'additional_information')
    // The same label on two lines: join the values instead of losing one.
    if (result[key]) {
      result[key] = { ...result[key], value: `${result[key].value}; ${value}` }
    } else {
      // An existing item keeps its source and highlight. A new one has none, and starts as "please verify".
      result[key] = previous[key]
        ? { ...previous[key], value }
        : { value, source_text: null, confidence: 'review', bbox: null, page: 1 }
    }
  }
  return result
}
