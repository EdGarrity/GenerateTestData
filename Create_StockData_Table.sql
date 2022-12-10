USE [SOS]
GO

/****** Object:  Table [dbo].[StockData]    Script Date: 12/10/2022 4:49:10 PM ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

CREATE TABLE [dbo].[StockData](
	[Stock] [varchar](11) NULL
	[Date] [datetime] NULL,
	[Open] [float] NULL,
	[High] [float] NULL,
	[Low] [float] NULL,
	[Close] [float] NULL,
	[Adj Close] [float] NULL,
	[Volume] [bigint] NULL,
	[Adj_Open] [float] NULL,
	[Adj_High] [float] NULL,
	[Adj_Low] [float] NULL,
	[Adj_Volume] [float] NULL,
	[Norm_Adj_Close] [float] NULL,
	[Norm_Adj_High] [float] NULL,
	[Norm_Adj_Low] [float] NULL,
	[Norm_Adj_Open] [float] NULL,
	[Norm_Adj_Volume] [float] NULL
) ON [PRIMARY]
GO


