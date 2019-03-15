.PHONY: deps
deps:
	@GO111MODULE=on go mod vendor

.PHONY: test
test:
	@go test -v ./... && echo "ALL PASS" || echo "FAILURE"
