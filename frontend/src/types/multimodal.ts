export type SensorEntryType =
  | 'wind_speed'
  | 'methane'
  | 'co'
  | 'temperature'
  | 'oxygen'
  | 'custom'

export type SensorEntry = {
  type: SensorEntryType
  label: string
  value: number
  unit: string
  location?: string
  timestamp?: string
  thresholdRef?: string
}

export type SensorData = {
  entries: SensorEntry[]
  location: string
  source: 'manual' | 'csv'
  rawCsv?: string
}

export type ChatMessageImage = {
  id: string
  name: string
  url: string
  size: number
  mimeType: string
  createdAt?: string
}

export type DraftImage = {
  id: string
  file: File
  preview: string
}
