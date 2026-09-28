// ---------------------------------------------------------------------------
// Reference infrastructure for azure-foundry-policy-agent.
//
// REFERENCE TEMPLATE: it is not deployed by CI and has not been deployed to a
// live subscription by the author. Review SKUs, regions, model availability and
// role definition IDs for your tenant before use.
//
// Creates:
//   - Azure AI Foundry resource (AIServices account) + Foundry project
//   - Azure OpenAI model deployments (chat + embeddings) on that resource
//   - Azure AI Search (semantic ranker enabled, keys disabled)
//   - Azure AI Content Safety
//   - Key Vault (RBAC mode, purge protection)
//   - Log Analytics workspace + workspace-based Application Insights
//   - A user-assigned managed identity for the app, with least-privilege roles
//
// Deploy:
//   az deployment group create -g <rg> -f infra/main.bicep -p infra/main.parameters.json
// ---------------------------------------------------------------------------

targetScope = 'resourceGroup'

@description('Short prefix for resource names (lowercase letters and digits).')
@minLength(3)
@maxLength(12)
param namePrefix string = 'policyagent'

@description('Region for all resources. Choose one where the models below are available and that meets your data-residency requirements.')
param location string = resourceGroup().location

@description('Deployment type for model deployments. "Standard" keeps processing in the chosen region; "DataZoneStandard"/"GlobalStandard" may process data outside it.')
@allowed([
  'Standard'
  'DataZoneStandard'
  'GlobalStandard'
])
param modelDeploymentSku string = 'Standard'

param chatModelName string = 'gpt-4o-mini'
param chatModelVersion string = '2024-07-18'
@description('Capacity in thousands of tokens per minute.')
param chatCapacity int = 30

param embeddingModelName string = 'text-embedding-3-small'
param embeddingModelVersion string = '1'
param embeddingCapacity int = 30

@allowed([
  'basic'
  'standard'
])
param searchSku string = 'basic'

@description('Set to Disabled once private endpoints are in place (recommended for production).')
@allowed([
  'Enabled'
  'Disabled'
])
param publicNetworkAccess string = 'Enabled'

@description('Optional object ID of the person or pipeline that runs the index builder. Leave empty to skip.')
param deployerPrincipalId string = ''

param tags object = {
  workload: 'policy-agent'
  classification: 'official'
}

var suffix = uniqueString(resourceGroup().id, namePrefix)
var names = {
  foundry: '${namePrefix}-ais-${suffix}'
  project: '${namePrefix}-project'
  search: '${namePrefix}-srch-${suffix}'
  contentSafety: '${namePrefix}-cs-${suffix}'
  keyVault: take('${namePrefix}kv${suffix}', 24)
  logAnalytics: '${namePrefix}-log-${suffix}'
  appInsights: '${namePrefix}-appi-${suffix}'
  identity: '${namePrefix}-id-${suffix}'
}

// Built-in role definition IDs (verify in your tenant: az role definition list --name "<role>").
var roles = {
  cognitiveServicesOpenAIUser: '5e0bd9bd-7b93-4f28-af87-19fc36ad61bd'
  cognitiveServicesUser: 'a97b65f3-24c7-4388-baec-2e87135dc908'
  azureAIUser: '53ca6127-db72-4b80-b1b0-d745d6d5456d'
  searchIndexDataReader: '1407120a-92aa-4202-b7e9-c0e197c71c8f'
  searchIndexDataContributor: '8ebe5a00-799e-43f5-93ac-243d3dce84a7'
  searchServiceContributor: '7ca78c08-252a-4471-8644-bb5ff32d4ba0'
  keyVaultSecretsUser: '4633458b-17de-408a-b874-0445c86b69e6'
  monitoringMetricsPublisher: '3913510d-42f4-4e42-8a64-420c390055eb'
}

// ----------------------------- Observability -------------------------------

resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: names.logAnalytics
  location: location
  tags: tags
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
  }
}

resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: names.appInsights
  location: location
  tags: tags
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logAnalytics.id
    // Only Entra-authenticated telemetry is accepted; the app identity gets Monitoring Metrics Publisher.
    DisableLocalAuth: true
  }
}

// ----------------------------- Identity & secrets --------------------------

resource appIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: names.identity
  location: location
  tags: tags
}

resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: names.keyVault
  location: location
  tags: tags
  properties: {
    tenantId: subscription().tenantId
    sku: {
      family: 'A'
      name: 'standard'
    }
    enableRbacAuthorization: true
    enableSoftDelete: true
    softDeleteRetentionInDays: 90
    enablePurgeProtection: true
    publicNetworkAccess: publicNetworkAccess
  }
}

// ----------------------------- AI Foundry ----------------------------------

resource foundry 'Microsoft.CognitiveServices/accounts@2025-06-01' = {
  name: names.foundry
  location: location
  tags: tags
  kind: 'AIServices'
  sku: {
    name: 'S0'
  }
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    customSubDomainName: names.foundry
    allowProjectManagement: true
    // Keys disabled: callers must use Entra ID (managed identity).
    disableLocalAuth: true
    publicNetworkAccess: publicNetworkAccess
  }
}

resource project 'Microsoft.CognitiveServices/accounts/projects@2025-06-01' = {
  parent: foundry
  name: names.project
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    displayName: 'Policy Q&A agent'
    description: 'Grounded government policy assistant'
  }
}

resource chatDeployment 'Microsoft.CognitiveServices/accounts/deployments@2025-06-01' = {
  parent: foundry
  name: chatModelName
  sku: {
    name: modelDeploymentSku
    capacity: chatCapacity
  }
  properties: {
    model: {
      format: 'OpenAI'
      name: chatModelName
      version: chatModelVersion
    }
    // Default content filter policy stays on; a stricter custom policy can be attached here.
    raiPolicyName: 'Microsoft.DefaultV2'
  }
}

resource embeddingDeployment 'Microsoft.CognitiveServices/accounts/deployments@2025-06-01' = {
  parent: foundry
  name: embeddingModelName
  sku: {
    name: modelDeploymentSku
    capacity: embeddingCapacity
  }
  properties: {
    model: {
      format: 'OpenAI'
      name: embeddingModelName
      version: embeddingModelVersion
    }
  }
  // Deployments on one account must be created one at a time.
  dependsOn: [
    chatDeployment
  ]
}

resource contentSafety 'Microsoft.CognitiveServices/accounts@2025-06-01' = {
  name: names.contentSafety
  location: location
  tags: tags
  kind: 'ContentSafety'
  sku: {
    name: 'S0'
  }
  properties: {
    customSubDomainName: names.contentSafety
    disableLocalAuth: true
    publicNetworkAccess: publicNetworkAccess
  }
}

// ----------------------------- AI Search -----------------------------------

resource search 'Microsoft.Search/searchServices@2023-11-01' = {
  name: names.search
  location: location
  tags: tags
  sku: {
    name: searchSku
  }
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    replicaCount: 1
    partitionCount: 1
    hostingMode: 'default'
    semanticSearch: 'free'
    disableLocalAuth: true
    publicNetworkAccess: toLower(publicNetworkAccess)
  }
}

// ----------------------------- Diagnostics ---------------------------------

resource foundryDiagnostics 'Microsoft.Insights/diagnosticSettings@2021-05-01-preview' = {
  name: 'to-log-analytics'
  scope: foundry
  properties: {
    workspaceId: logAnalytics.id
    logs: [
      {
        categoryGroup: 'allLogs'
        enabled: true
      }
    ]
    metrics: [
      {
        category: 'AllMetrics'
        enabled: true
      }
    ]
  }
}

resource searchDiagnostics 'Microsoft.Insights/diagnosticSettings@2021-05-01-preview' = {
  name: 'to-log-analytics'
  scope: search
  properties: {
    workspaceId: logAnalytics.id
    logs: [
      {
        categoryGroup: 'allLogs'
        enabled: true
      }
    ]
    metrics: [
      {
        category: 'AllMetrics'
        enabled: true
      }
    ]
  }
}

// ----------------------------- Role assignments (app identity) -------------

resource appOpenAIUser 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(foundry.id, appIdentity.id, roles.cognitiveServicesOpenAIUser)
  scope: foundry
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.cognitiveServicesOpenAIUser)
    principalId: appIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

resource appAIUser 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(foundry.id, appIdentity.id, roles.azureAIUser)
  scope: foundry
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.azureAIUser)
    principalId: appIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

resource appContentSafetyUser 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(contentSafety.id, appIdentity.id, roles.cognitiveServicesUser)
  scope: contentSafety
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.cognitiveServicesUser)
    principalId: appIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

resource appSearchReader 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(search.id, appIdentity.id, roles.searchIndexDataReader)
  scope: search
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.searchIndexDataReader)
    principalId: appIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

resource appKeyVaultSecretsUser 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(keyVault.id, appIdentity.id, roles.keyVaultSecretsUser)
  scope: keyVault
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.keyVaultSecretsUser)
    principalId: appIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

resource appTelemetryPublisher 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(appInsights.id, appIdentity.id, roles.monitoringMetricsPublisher)
  scope: appInsights
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.monitoringMetricsPublisher)
    principalId: appIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// ----------------------------- Role assignments (index builder) ------------

resource deployerSearchContributor 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(deployerPrincipalId)) {
  name: guid(search.id, deployerPrincipalId, roles.searchServiceContributor)
  scope: search
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.searchServiceContributor)
    principalId: deployerPrincipalId
  }
}

resource deployerSearchDataContributor 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(deployerPrincipalId)) {
  name: guid(search.id, deployerPrincipalId, roles.searchIndexDataContributor)
  scope: search
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.searchIndexDataContributor)
    principalId: deployerPrincipalId
  }
}

resource deployerOpenAIUser 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(deployerPrincipalId)) {
  name: guid(foundry.id, deployerPrincipalId, roles.cognitiveServicesOpenAIUser)
  scope: foundry
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.cognitiveServicesOpenAIUser)
    principalId: deployerPrincipalId
  }
}

// ----------------------------- Outputs (no secrets) ------------------------

output AZURE_AI_PROJECT_ENDPOINT string = 'https://${names.foundry}.services.ai.azure.com/api/projects/${project.name}'
output AZURE_OPENAI_ENDPOINT string = foundry.properties.endpoint
output AZURE_OPENAI_CHAT_DEPLOYMENT string = chatDeployment.name
output AZURE_OPENAI_EMBEDDING_DEPLOYMENT string = embeddingDeployment.name
output AZURE_SEARCH_ENDPOINT string = 'https://${search.name}.search.windows.net'
output AZURE_CONTENT_SAFETY_ENDPOINT string = contentSafety.properties.endpoint
output KEY_VAULT_URI string = keyVault.properties.vaultUri
output APP_IDENTITY_CLIENT_ID string = appIdentity.properties.clientId
output APP_IDENTITY_RESOURCE_ID string = appIdentity.id
output APPLICATIONINSIGHTS_CONNECTION_STRING string = appInsights.properties.ConnectionString
