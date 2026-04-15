import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { uploadDocument } from '../services/api'

const ACCEPTED_TYPES = {
  'application/pdf': ['.pdf'],
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
  'text/plain': ['.txt'],
}

const MAX_FILENAME_LENGTH = 28

function truncateFilename(name) {
  if (name.length <= MAX_FILENAME_LENGTH) return name
  const ext = name.slice(name.lastIndexOf('.'))
  const stem = name.slice(0, MAX_FILENAME_LENGTH - ext.length - 3)
  return `${stem}...${ext}`
}

function FileIcon() {
  return (
    <svg className="w-4 h-4 flex-shrink-0 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
        d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
    </svg>
  )
}

function UploadIcon() {
  return (
    <svg className="w-8 h-8 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
        d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
    </svg>
  )
}

function CheckBadge() {
  return (
    <svg className="w-4 h-4 flex-shrink-0 text-green-400" fill="currentColor" viewBox="0 0 20 20">
      <path fillRule="evenodd"
        d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
        clipRule="evenodd" />
    </svg>
  )
}

// Status pill shown while a file is uploading or has errored.
function UploadStatus({ status, error }) {
  if (status === 'uploading') {
    return (
      <div className="mt-3 flex items-center gap-2 text-xs text-blue-400">
        <svg className="w-3.5 h-3.5 animate-spin" fill="none" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
        </svg>
        Uploading…
      </div>
    )
  }
  if (status === 'error') {
    return (
      <p className="mt-2 text-xs text-red-400 break-words">{error}</p>
    )
  }
  return null
}

export default function DocumentSidebar({
  uploadedDocs,
  onUploadSuccess,
  onDocumentSelect,
  selectedDocs,
}) {
  const [uploadStatus, setUploadStatus] = useState('idle') // idle | uploading | success | error
  const [uploadError, setUploadError] = useState(null)

  const onDrop = useCallback(async (acceptedFiles) => {
    if (acceptedFiles.length === 0) return

    setUploadStatus('uploading')
    setUploadError(null)

    // Upload files sequentially so progress feedback is clear.
    for (const file of acceptedFiles) {
      try {
        await uploadDocument(file)
        onUploadSuccess(file.name)
      } catch (err) {
        setUploadStatus('error')
        setUploadError(err.message)
        return
      }
    }

    setUploadStatus('success')
    // Reset to idle after a short confirmation delay.
    setTimeout(() => setUploadStatus('idle'), 2000)
  }, [onUploadSuccess])

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept: ACCEPTED_TYPES,
    disabled: uploadStatus === 'uploading',
  })

  const dropzoneBorder =
    isDragReject ? 'border-red-500 bg-red-950/30' :
    isDragActive  ? 'border-blue-500 bg-blue-950/30' :
    uploadStatus === 'success' ? 'border-green-600 bg-green-950/20' :
    'border-gray-700 hover:border-gray-500'

  return (
    <aside className="w-64 min-h-screen bg-gray-900 border-r border-gray-800 flex flex-col p-4 gap-6">
      {/* Header */}
      <div>
        <h2 className="text-white font-semibold text-sm tracking-wide uppercase">
          Documents
        </h2>
        <p className="text-gray-500 text-xs mt-0.5">PDF · DOCX · TXT</p>
      </div>

      {/* Drop zone */}
      <div
        {...getRootProps()}
        className={`flex flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed p-5 cursor-pointer transition-colors ${dropzoneBorder}`}
      >
        <input {...getInputProps()} />
        <UploadIcon />
        <p className="text-center text-xs text-gray-400 leading-relaxed">
          {isDragActive
            ? isDragReject
              ? 'Unsupported file type'
              : 'Drop to upload'
            : uploadStatus === 'success'
            ? '✓ Uploaded successfully'
            : <>Drag & drop or <span className="text-white underline">browse</span></>}
        </p>
        <UploadStatus status={uploadStatus} error={uploadError} />
      </div>

      {/* Document list */}
      {uploadedDocs.length > 0 && (
        <div className="flex flex-col gap-1">
          <p className="text-gray-500 text-xs uppercase tracking-wide mb-1">
            Ingested ({uploadedDocs.length})
          </p>
          {uploadedDocs.map((doc) => {
            const isSelected = selectedDocs.includes(doc)
            return (
              <button
                key={doc}
                onClick={() => onDocumentSelect(doc)}
                className={`flex items-center gap-2.5 w-full rounded-md px-2.5 py-2 text-left transition-colors
                  ${isSelected
                    ? 'bg-gray-700 text-white'
                    : 'text-gray-300 hover:bg-gray-800 hover:text-white'
                  }`}
              >
                {/* Checkbox */}
                <span className={`w-4 h-4 flex-shrink-0 rounded border transition-colors flex items-center justify-center
                  ${isSelected ? 'bg-blue-500 border-blue-500' : 'border-gray-600'}`}>
                  {isSelected && (
                    <svg className="w-2.5 h-2.5 text-white" fill="currentColor" viewBox="0 0 12 12">
                      <path d="M10 3L5 8.5 2 5.5l-1 1L5 10.5l6-7.5-1-1z" />
                    </svg>
                  )}
                </span>

                <FileIcon />

                <span className="text-xs flex-1 truncate" title={doc}>
                  {truncateFilename(doc)}
                </span>

                <CheckBadge />
              </button>
            )
          })}
        </div>
      )}

      {/* Empty state */}
      {uploadedDocs.length === 0 && (
        <p className="text-gray-600 text-xs text-center mt-2">
          No documents yet.<br />Upload one to get started.
        </p>
      )}
    </aside>
  )
}
