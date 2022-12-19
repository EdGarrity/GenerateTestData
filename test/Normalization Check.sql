-- Checks the Normalization Logic.  Results should be zero if good

SELECT SUM([Normalization Check_Data]) as [Normalization Check]
FROM
(
--   SELECT ([Adj_Open]/[Adj Close]-[Norm_Adj_Open]/[Norm_Adj_Close])/([Adj_Open]/[Adj Close])*100 AS [Normalization Check_Data]
   SELECT ([Adj_Open]/[Adj Close]-[Norm_Adj_Open]/[Norm_Adj_Close]) AS [Normalization Check_Data]
   FROM [SOS].[dbo].[StockData]
--   where [Norm_Adj_Close]!=0
) AS X
