import axios from 'axios'

const API_BASE = '/api/v1'

const apiClient = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Global response interceptor — normalises all API errors into a single
// Error with a user-readable message so callers never see raw axios noise.
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    const status = error.response?.status
    const detail = error.response?.data?.detail

    let message

    if (!error.response) {
      // Network failure or server unreachable.
      message = 'Cannot reach the server. Make sure the backend is running.'
    } else if (detail) {
      // FastAPI HTTPException detail string.
      message = typeof detail === 'string' ? detail : JSON.stringify(detail)
    } else if (status === 400) {
      message = 'Bad request. Please check your input.'
    } else if (status === 413) {
      message = 'File is too large. Maximum allowed size is 50 MB.'
    } else if (status === 422) {
      message = 'Validation error. Please check the request format.'
    } else if (status === 500) {
      message = 'Server error. Please try again or check the backend logs.'
    } else {
      message = `Unexpected error (HTTP ${status ?? 'unknown'}).`
    }

    return Promise.reject(new Error(message))
  }
)

/**
 * Upload a document file for ingestion.
 *
 * @param {File} file - The file object from an <input> or drag-and-drop event.
 * @returns {Promise<{status: string, filename: string, total_chunks: number, message: string}>}
 */
export async function uploadDocument(file) {
  const formData = new FormData()
  formData.append('file', file)

  const response = await apiClient.post('/documents/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return response.data
}

/**
 * Ask a question and receive a grounded answer with citations.
 *
 * @param {string} query - Natural-language question.
 * @param {string} [collectionName="documents"] - Qdrant collection to search.
 * @param {number} [topN=5] - Number of chunks to use as LLM context.
 * @returns {Promise<Object>} Full QueryResponse from the backend.
 */
export async function askQuestion(query, collectionName = 'documents', topN = 5) {
  const response = await apiClient.post('/query/ask', {
    query,
    collection_name: collectionName,
    top_n: topN,
  })
  return response.data
}

/**
 * Run a raw hybrid search without LLM processing.
 * Useful for inspecting retrieval quality.
 *
 * @param {string} query - Natural-language question.
 * @param {string} [collectionName="documents"] - Qdrant collection to search.
 * @returns {Promise<Object>} SearchResponse with ranked chunk list.
 */
export async function searchChunks(query, collectionName = 'documents') {
  const response = await apiClient.post('/query/search', {
    query,
    collection_name: collectionName,
  })
  return response.data
}

/**
 * Compare answers to the same question across multiple documents.
 *
 * @param {string} query - Natural-language question.
 * @param {string[]} filenames - Array of filenames to compare.
 * @returns {Promise<Object>} CompareResponse with per-document answers.
 */
export async function compareDocuments(query, filenames) {
  const response = await apiClient.post('/query/compare', {
    query,
    filenames,
    collection_name: 'documents',
  })
  return response.data
}
